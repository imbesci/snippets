"""
Near-Realtime Activity Poller Service
------------------------------------
A production-ready asyncio service that continuously polls up to ~400 items
for status updates with adaptive cadence, cautious downstream usage, and a
fully-configurable setup.

Key Features
============
- Initial batch load of activities (activity_date, loan_alias)
- Daily refresh of activity list (configurable time & interval)
- Two-tier polling cadence:
  * High frequency (default 60s) for items in [-1 day, +3 days] window around today
  * Scaled slower cadence up to 180s for items outside that window
- Auto-promotion to high frequency if a slow-polled item returns
  response.json().loanRepricing.statusCode != null
- Global, per-host Token Bucket rate limiting + concurrency limits
- Exponential backoff with jitter and circuit-breaker style cooldowns
- Pluggable API client & Repository layer for database
- Structured logging
- Graceful shutdown
- All parameters configurable via YAML and environment overrides

Quick Start
===========
1) Create a config file (config.yaml) using the template below.
2) `pip install aiohttp pydantic pyyaml python-dateutil` (plus your DB deps)
3) Run: `python service.py`

Example config.yaml
===================
service:
  log_level: INFO
  # How often to poll when in the high-frequency window
  high_poll_seconds: 60
  # The maximum interval when outside the window
  max_poll_seconds: 180
  # Window bounds relative to 'today' (can be negative/positive)
  window_before_days: 1
  window_after_days: 3
  # Daily refresh of activities (cron-like simple HH:MM, 24h)
  daily_refresh_time: "02:15"
  daily_refresh_jitter_seconds: 120
  # Number of workers concurrently performing polls
  max_concurrent_polls: 64
  # Per-activity jitter to de-sync schedules
  per_activity_jitter_seconds: 7
  # Circuit-breaker cooldown when repeated failures
  circuit_breaker_cooldown_seconds: 30
  # Max retries per poll attempt
  max_retries: 3
  # Base backoff seconds; jitter added each retry
  retry_backoff_base_seconds: 0.5
  # Max backoff seconds cap
  retry_backoff_max_seconds: 5.0

limiters:
  # Token bucket per host (or "global")
  # refill_rate: tokens per second, capacity: bucket size
  - name: global
    match: "*"
    refill_rate: 10.0   # ~10 rps
    capacity: 20
  - name: downstream_api
    match: "api.example.com"
    refill_rate: 5.0    # ~5 rps for that host
    capacity: 10

api:
  base_url: "https://api.example.com"
  timeout_seconds: 15
  # Optional auth headers or token
  headers:
    Authorization: "Bearer ${API_TOKEN}"

repository:
  # Choose one: "noop", "postgres", "mysql", "sqlite"
  kind: "noop"
  # Add DSN or connection configs for your real DB implementation
  dsn: "postgresql+asyncpg://user:pass@localhost:5432/mydb"

activities_source:
  # How we load/refresh the batch of activities
  # Choose one: "repository", "http", "file"
  kind: "repository"
  # For "http", specify URL; for "file", specify path
  http_url: "https://example.com/activities"
  file_path: "./activities.json"

"""
from __future__ import annotations

import asyncio
import dataclasses
import heapq
import json
import logging
import math
import os
import random
import signal
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import aiohttp
import yaml
from dateutil import tz

# ---------------------------- Utilities & Config ---------------------------- #

@dataclass
class ServiceConfig:
    log_level: str = "INFO"
    high_poll_seconds: int = 60
    max_poll_seconds: int = 180
    window_before_days: int = 1
    window_after_days: int = 3
    daily_refresh_time: str = "02:15"  # HH:MM 24h
    daily_refresh_jitter_seconds: int = 120
    max_concurrent_polls: int = 64
    per_activity_jitter_seconds: int = 7
    circuit_breaker_cooldown_seconds: int = 30
    max_retries: int = 3
    retry_backoff_base_seconds: float = 0.5
    retry_backoff_max_seconds: float = 5.0

@dataclass
class LimiterRule:
    name: str
    match: str  # host or "*"
    refill_rate: float  # tokens per second
    capacity: int

@dataclass
class ApiConfig:
    base_url: str
    timeout_seconds: int = 15
    headers: Dict[str, str] = field(default_factory=dict)

@dataclass
class RepoConfig:
    kind: str = "noop"
    dsn: str = ""

@dataclass
class ActivitiesSourceConfig:
    kind: str = "repository"  # or "http" or "file"
    http_url: str = ""
    file_path: str = ""

@dataclass
class Config:
    service: ServiceConfig
    limiters: List[LimiterRule]
    api: ApiConfig
    repository: RepoConfig
    activities_source: ActivitiesSourceConfig

    @staticmethod
    def load(path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        # Expand environment variables in strings
        def expand(obj):
            if isinstance(obj, str):
                return os.path.expandvars(obj)
            if isinstance(obj, list):
                return [expand(x) for x in obj]
            if isinstance(obj, dict):
                return {k: expand(v) for k, v in obj.items()}
            return obj
        raw = expand(raw)
        svc = ServiceConfig(**raw.get("service", {}))
        lims = [LimiterRule(**d) for d in raw.get("limiters", [])]
        api = ApiConfig(**raw.get("api", {}))
        repo = RepoConfig(**raw.get("repository", {}))
        src = ActivitiesSourceConfig(**raw.get("activities_source", {}))
        return Config(svc, lims, api, repo, src)

# ------------------------------- Domain Types ------------------------------ #

@dataclass(order=True)
class ScheduledItem:
    next_run_at: float
    loan_alias: str = field(compare=False)
    activity_date: date = field(compare=False)
    priority: int = field(default=0, compare=False)

@dataclass
class Activity:
    loan_alias: str
    activity_date: date

# --------------------------- Rate Limiter (Token) -------------------------- #

class TokenBucket:
    def __init__(self, refill_rate: float, capacity: int):
        self.refill_rate = float(refill_rate)
        self.capacity = int(capacity)
        self.tokens = float(capacity)
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0):
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self.last_refill
                self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
                self.last_refill = now
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return
                need = tokens - self.tokens
                wait_time = need / self.refill_rate if self.refill_rate > 0 else 0.1
            await asyncio.sleep(wait_time)

# ------------------------------- Repositories ------------------------------ #

class Repository:
    async def upsert_status(self, loan_alias: str, status: Mapping[str, Any]) -> None:
        raise NotImplementedError

    async def load_activities(self) -> List[Activity]:
        raise NotImplementedError

class NoopRepository(Repository):
    def __init__(self):
        self._store: Dict[str, Mapping[str, Any]] = {}
        self._activities: List[Activity] = []

    async def upsert_status(self, loan_alias: str, status: Mapping[str, Any]) -> None:
        self._store[loan_alias] = status

    async def load_activities(self) -> List[Activity]:
        return list(self._activities)

    def set_activities(self, acts: List[Activity]):
        self._activities = acts

# TODO: Implement concrete repositories (e.g., Postgres/SQLAlchemy) as needed

# ------------------------------- Activities IO ----------------------------- #

class ActivitiesSource:
    def __init__(self, cfg: ActivitiesSourceConfig, repo: Repository, session: aiohttp.ClientSession):
        self.cfg = cfg
        self.repo = repo
        self.session = session

    async def fetch(self) -> List[Activity]:
        kind = self.cfg.kind.lower()
        if kind == "repository":
            return await self.repo.load_activities()
        elif kind == "http":
            async with self.session.get(self.cfg.http_url, timeout=30) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return [Activity(loan_alias=x["loan_alias"], activity_date=_parse_date(x["activity_date"])) for x in data]
        elif kind == "file":
            with open(self.cfg.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [Activity(loan_alias=x["loan_alias"], activity_date=_parse_date(x["activity_date"])) for x in data]
        else:
            raise ValueError(f"Unsupported activities_source.kind: {self.cfg.kind}")

# --------------------------------- API Client ------------------------------ #

class ApiClient:
    def __init__(self, cfg: ApiConfig, limiters: List[LimiterRule]):
        timeout = aiohttp.ClientTimeout(total=cfg.timeout_seconds)
        self.session = aiohttp.ClientSession(timeout=timeout, headers=cfg.headers)
        self.cfg = cfg
        self.buckets: Dict[str, TokenBucket] = {}
        for rule in limiters:
            # Create a bucket keyed by rule.match
            self.buckets[rule.match] = TokenBucket(rule.refill_rate, rule.capacity)
        # Fallback global bucket if none provided
        if "*" not in self.buckets:
            self.buckets["*"] = TokenBucket(refill_rate=10.0, capacity=20)

    async def close(self):
        await self.session.close()

    def _bucket_for_host(self, host: str) -> TokenBucket:
        return self.buckets.get(host, self.buckets.get("*"))

    async def get_status(self, loan_alias: str) -> Mapping[str, Any]:
        # Example endpoint; adjust path or query per your API
        url = f"{self.cfg.base_url}/loans/{loan_alias}/repricing"
        host = aiohttp.helpers.urlparse(url).hostname or "*"
        bucket = self._bucket_for_host(host)
        await bucket.acquire(1)
        async with self.session.get(url) as resp:
            resp.raise_for_status()
            return await resp.json()

# --------------------------- Scheduling & Cadence --------------------------- #

def _parse_date(s: str) -> date:
    return datetime.fromisoformat(s).date()

def now_ts() -> float:
    return time.monotonic()

def today_local() -> date:
    # Use local timezone (or make configurable)
    return datetime.now(tz=tz.tzlocal()).date()

class CadencePlanner:
    def __init__(self, cfg: ServiceConfig):
        self.cfg = cfg

    def interval_for(self, activity_date: date, promoted: bool) -> float:
        """Return seconds until next poll for an activity.
        - If promoted==True => high cadence regardless of date
        - Else: if within [today - before, today + after] => high cadence
                otherwise scale up to max_poll_seconds based on distance.
        """
        if promoted:
            return float(self.cfg.high_poll_seconds)
        today = today_local()
        in_window_start = today - timedelta(days=self.cfg.window_before_days)
        in_window_end = today + timedelta(days=self.cfg.window_after_days)
        if in_window_start <= activity_date <= in_window_end:
            return float(self.cfg.high_poll_seconds)
        # Outside window: scale to max based on distance in days from nearest boundary
        if activity_date < in_window_start:
            delta_days = (in_window_start - activity_date).days
        else:
            delta_days = (activity_date - in_window_end).days
        # Linear scale from high -> max; clamp
        span = max(1, self.cfg.window_after_days + self.cfg.window_before_days)
        # Map delta_days starting at 1 day out to max
        frac = min(1.0, delta_days / span)
        interval = self.cfg.high_poll_seconds + frac * (self.cfg.max_poll_seconds - self.cfg.high_poll_seconds)
        return float(min(self.cfg.max_poll_seconds, max(self.cfg.high_poll_seconds, interval)))

# ------------------------------ Poll Orchestration ------------------------- #

class Poller:
    def __init__(self, cfg: Config, repo: Repository, api: ApiClient):
        self.cfg = cfg
        self.repo = repo
        self.api = api
        self.planner = CadencePlanner(cfg.service)
        self.queue: List[ScheduledItem] = []  # min-heap by next_run_at
        self.promoted: set[str] = set()
        self.inflight = 0
        self.max_concurrent = cfg.service.max_concurrent_polls
        self._shutdown = asyncio.Event()
        self._wakeup = asyncio.Event()
        self._sema = asyncio.Semaphore(self.max_concurrent)
        self._failures: Dict[str, int] = defaultdict(int)

    def _push_heap(self, item: ScheduledItem):
        heapq.heappush(self.queue, item)

    def _pop_heap(self) -> ScheduledItem:
        return heapq.heappop(self.queue)

    def _schedule_initial(self, activities: Iterable[Activity]):
        base_jitter = self.cfg.service.per_activity_jitter_seconds
        now = now_ts()
        for a in activities:
            jitter = random.uniform(0, base_jitter)
            self._push_heap(ScheduledItem(next_run_at=now + jitter, loan_alias=a.loan_alias, activity_date=a.activity_date))

    def _reschedule(self, item: ScheduledItem, immediate: bool = False):
        interval = self.planner.interval_for(item.activity_date, item.loan_alias in self.promoted)
        if immediate:
            delay = random.uniform(0, self.cfg.service.per_activity_jitter_seconds)
        else:
            delay = interval + random.uniform(0, self.cfg.service.per_activity_jitter_seconds)
        item.next_run_at = now_ts() + delay
        self._push_heap(item)

    async def _do_poll_once(self, item: ScheduledItem):
        loan_alias = item.loan_alias
        tries = 0
        while True:
            try:
                result = await self.api.get_status(loan_alias)
                # Persist to DB
                await self.repo.upsert_status(loan_alias, result)
                # Auto-promotion: response.json().loanRepricing.statusCode != null
                status_code = None
                try:
                    status_code = result.get("loanRepricing", {}).get("statusCode")
                except Exception:
                    status_code = None
                if status_code is not None:
                    self.promoted.add(loan_alias)
                self._failures[loan_alias] = 0
                return
            except Exception as e:
                tries += 1
                self._failures[loan_alias] += 1
                logging.warning("poll error for %s (try %s): %s", loan_alias, tries, e)
                if tries >= self.cfg.service.max_retries:
                    # apply circuit-breaker style cooldown
                    await asyncio.sleep(self.cfg.service.circuit_breaker_cooldown_seconds)
                    return
                backoff = min(self.cfg.service.retry_backoff_max_seconds,
                              self.cfg.service.retry_backoff_base_seconds * (2 ** (tries - 1)))
                backoff += random.uniform(0, 0.5)
                await asyncio.sleep(backoff)

    async def _worker(self, item: ScheduledItem):
        async with self._sema:
            await self._do_poll_once(item)
            self._reschedule(item)

    async def run(self, initial_activities: List[Activity]):
        self._schedule_initial(initial_activities)
        logging.info("Scheduled %d activities", len(self.queue))
        while not self._shutdown.is_set():
            if not self.queue:
                await asyncio.sleep(0.5)
                continue
            next_item = self.queue[0]
            delay = max(0.0, next_item.next_run_at - now_ts())
            # Wakeup early if requested (e.g., activities refresh)
            try:
                await asyncio.wait_for(self._wakeup.wait(), timeout=delay)
                self._wakeup.clear()
                continue  # re-evaluate heap top immediately
            except asyncio.TimeoutError:
                pass
            now = now_ts()
            # Pop and dispatch all due items
            batch: List[ScheduledItem] = []
            while self.queue and self.queue[0].next_run_at <= now:
                batch.append(self._pop_heap())
            # Dispatch batch
            tasks = [asyncio.create_task(self._worker(it)) for it in batch]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    def promote(self, loan_alias: str):
        self.promoted.add(loan_alias)
        self._wakeup.set()

    def replace_activities(self, new_activities: List[Activity]):
        # Rebuild heap preserving promotions for existing aliases
        promos = set(self.promoted)
        self.queue.clear()
        self.promoted = promos.intersection({a.loan_alias for a in new_activities})
        self._schedule_initial(new_activities)
        self._wakeup.set()

    async def shutdown(self):
        self._shutdown.set()
        self._wakeup.set()

# ------------------------------ Daily Refresh ------------------------------ #

class DailyRefresher:
    def __init__(self, cfg: ServiceConfig, source: ActivitiesSource, poller: Poller):
        self.cfg = cfg
        self.source = source
        self.poller = poller
        self._shutdown = asyncio.Event()

    def _next_run_at(self) -> float:
        # Parse HH:MM in local time and compute next occurrence
        hh, mm = map(int, self.cfg.daily_refresh_time.split(":"))
        now_local = datetime.now(tz=tz.tzlocal())
        scheduled = now_local.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if scheduled <= now_local:
            scheduled += timedelta(days=1)
        jitter = random.uniform(0, self.cfg.daily_refresh_jitter_seconds)
        scheduled += timedelta(seconds=jitter)
        return scheduled.timestamp()

    async def run(self):
        while not self._shutdown.is_set():
            run_at = self._next_run_at()
            now_ts_wall = datetime.now(tz=tz.tzlocal()).timestamp()
            await asyncio.sleep(max(0, run_at - now_ts_wall))
            try:
                acts = await self.source.fetch()
                self.poller.replace_activities(acts)
                logging.info("Daily activities refreshed: %d items", len(acts))
            except Exception as e:
                logging.error("Daily refresh failed: %s", e)

    async def shutdown(self):
        self._shutdown.set()

# ------------------------------- App Assembly ------------------------------ #

async def build_repository(cfg: RepoConfig) -> Repository:
    kind = cfg.kind.lower()
    if kind == "noop":
        return NoopRepository()
    # TODO: Implement real repositories; return appropriate instance
    return NoopRepository()

async def initial_activities(source: ActivitiesSource, repo: Repository) -> List[Activity]:
    acts = await source.fetch()
    # If using Noop repo and nothing returned, add a demo set
    if isinstance(repo, NoopRepository) and not acts:
        today = today_local()
        demo = []
        for i in range(1, 401):
            # Spread dates around the window and outside
            d = today + timedelta(days=random.randint(-3, 10))
            demo.append(Activity(loan_alias=f"LN{i:04d}", activity_date=d))
        repo.set_activities(demo)
        acts = demo
    return acts

async def main(config_path: str = "config.yaml"):
    cfg = Config.load(config_path)
    logging.basicConfig(level=getattr(logging, cfg.service.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(message)s")

    repo = await build_repository(cfg.repository)
    api = ApiClient(cfg.api, cfg.limiters)

    # ActivitiesSource needs an aiohttp session; reuse api.session for simplicity
    source = ActivitiesSource(cfg.activities_source, repo, api.session)

    acts = await initial_activities(source, repo)

    poller = Poller(cfg, repo, api)
    refresher = DailyRefresher(cfg.service, source, poller)

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _handle_signal():
        logging.info("Shutdown signal received")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_signal)
        except NotImplementedError:
            # Windows compatibility
            pass

    tasks = [
        asyncio.create_task(poller.run(acts), name="poller"),
        asyncio.create_task(refresher.run(), name="refresher"),
    ]

    await stop_event.wait()

    await refresher.shutdown()
    await poller.shutdown()

    for t in tasks:
        t.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await t

    await api.close()

# ----------------------------- Entrypoint ---------------------------------- #

if __name__ == "__main__":
    import contextlib
    import sys
    try:
        config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
        asyncio.run(main(config_path))
    except KeyboardInterrupt:
        pass

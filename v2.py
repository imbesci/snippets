#!/usr/bin/env python3
"""
Near-realtime status sync service.

- Polls activities with dynamic cadences (60s in-window; up to 180s out-of-window).
- Auto-promotes cadence to 60s if response has non-null loanRepricing.statusCode.
- Starts with a batch fetch; refreshes activities daily.
- Downstream-friendly: global token-bucket RPS limit, max concurrency, backoff+jitter.
- Syncs to DB by POSTing to your endpoint.
- Optional Excel report on Ctrl+C (enabled via flag). No memory growth if disabled.

Python 3.10+
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import datetime as dt
import json
import logging
import math
import os
import random
import signal
import sys
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

# Optional dependencies (only used if --report is enabled):
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # We guard usage so the service still runs if pandas isn't installed.


# ---------------------------
# Configuration
# ---------------------------

@dataclasses.dataclass
class Config:
    # --- Activity fetch & refresh ---
    activities_bootstrap_url: str = os.getenv("ACTIVITIES_BOOTSTRAP_URL", "https://example.com/api/activities")
    activities_bootstrap_method: str = os.getenv("ACTIVITIES_BOOTSTRAP_METHOD", "GET")  # GET or POST
    activities_refresh_hour_local: int = int(os.getenv("ACTIVITIES_REFRESH_HOUR_LOCAL", "2"))  # refresh daily at 2am local
    activities_page_size: int = int(os.getenv("ACTIVITIES_PAGE_SIZE", "500"))  # if paginating

    # --- Polling cadences (seconds) ---
    high_interval_s: int = int(os.getenv("HIGH_INTERVAL_S", "60"))          # in-window
    max_slow_interval_s: int = int(os.getenv("MAX_SLOW_INTERVAL_S", "180")) # out-of-window cap

    # --- Window controls (relative days to 'today') ---
    window_before_days: int = int(os.getenv("WINDOW_BEFORE_DAYS", "1"))   # [CUSTOMIZABLE PARAM 1 DAY BEFORE] default 1
    window_after_days: int = int(os.getenv("WINDOW_AFTER_DAYS", "3"))     # [CUSTOMIZABLE PARAM 3 DAYS AFTER] default 3

    # --- Downstream protection ---
    max_concurrency: int = int(os.getenv("MAX_CONCURRENCY", "40"))        # max simultaneous requests
    rps_limit: float = float(os.getenv("RPS_LIMIT", "20"))                 # steady tokens per second
    burst: int = int(os.getenv("RPS_BURST", "40"))                         # bucket size
    request_timeout_s: int = int(os.getenv("REQUEST_TIMEOUT_S", "15"))
    connect_timeout_s: int = int(os.getenv("CONNECT_TIMEOUT_S", "5"))
    retry_base_delay_s: float = float(os.getenv("RETRY_BASE_DELAY_S", "0.6"))
    retry_max_delay_s: float = float(os.getenv("RETRY_MAX_DELAY_S", "8"))
    retry_attempts: int = int(os.getenv("RETRY_ATTEMPTS", "4"))

    # --- Endpoints ---
    status_poll_url: str = os.getenv("STATUS_POLL_URL", "https://example.com/api/status")  # GET with ?loan_alias=...
    db_sync_post_url: str = os.getenv("DB_SYNC_POST_URL", "https://example.com/api/db/sync")  # POST JSON

    # --- Auth / headers ---
    api_key: Optional[str] = os.getenv("API_KEY")  # if needed
    extra_headers_json: Optional[str] = os.getenv("EXTRA_HEADERS_JSON")  # e.g. '{"X-Tenant":"acme"}'

    # --- Housekeeping ---
    activities_refresh_tz: str = os.getenv("ACTIVITIES_REFRESH_TZ", "local")  # "local" or "UTC"
    report_enabled: bool = os.getenv("REPORT_ENABLED", "false").lower() in {"1", "true", "yes"}
    report_path: str = os.getenv("REPORT_PATH", "sync_report.xlsx")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # --- Service identity ---
    service_name: str = os.getenv("SERVICE_NAME", "status-sync-service")

    def headers(self) -> Dict[str, str]:
        h = {
            "User-Agent": f"{self.service_name}/1.0",
            "Accept": "application/json",
        }
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        if self.extra_headers_json:
            with contextlib.suppress(Exception):
                h.update(json.loads(self.extra_headers_json))
        return h


# ---------------------------
# Logging
# ---------------------------

def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


logger = logging.getLogger("sync")


# ---------------------------
# Rate Limiting (Token Bucket)
# ---------------------------

class TokenBucket:
    """Simple async token bucket."""
    def __init__(self, rate: float, capacity: int) -> None:
        self.rate = rate
        self.capacity = capacity
        self.tokens = float(capacity)
        self.updated = asyncio.get_event_loop().time()
        self._lock = asyncio.Lock()

    async def take(self, tokens: float = 1.0) -> None:
        async with self._lock:
            now = asyncio.get_event_loop().time()
            elapsed = now - self.updated
            self.updated = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            if self.tokens >= tokens:
                self.tokens -= tokens
                return
            # need to wait
            need = tokens - self.tokens
            wait_s = need / self.rate if self.rate > 0 else 0.05
        await asyncio.sleep(wait_s)
        # Try again (tail recursion elided)
        await self.take(tokens)


# ---------------------------
# Activity model
# ---------------------------

@dataclasses.dataclass(slots=True)
class Activity:
    loan_alias: str
    activity_date: dt.date


# ---------------------------
# Report collector (optional)
# ---------------------------

@dataclasses.dataclass
class ReportItem:
    timestamp: dt.datetime
    loan_alias: str
    activity_date: dt.date
    attempt: int
    success: bool
    http_status: Optional[int]
    error: Optional[str]
    promoted_to_high: bool
    interval_used_s: int


class ReportCollector:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._items: List[ReportItem] = []

    def add(self, item: ReportItem) -> None:
        if not self.enabled:
            return
        self._items.append(item)

    def write_excel(self, cfg: Config) -> Optional[str]:
        if not self.enabled:
            return None
        if pd is None:
            logger.warning("Report enabled but pandas is not installed; writing CSV instead.")
            path = cfg.report_path.replace(".xlsx", ".csv")
            with open(path, "w", encoding="utf-8") as f:
                f.write("timestamp,loan_alias,activity_date,attempt,success,http_status,error,promoted_to_high,interval_used_s\n")
                for it in self._items:
                    f.write(f"{it.timestamp.isoformat()},{it.loan_alias},{it.activity_date.isoformat()},{it.attempt},{int(it.success)},{it.http_status or ''},{(it.error or '').replace(',', ';')},{int(it.promoted_to_high)},{it.interval_used_s}\n")
            return path

        df = pd.DataFrame([dataclasses.asdict(x) for x in self._items])
        # Add a sheet with config parameters for provenance
        cfg_df = pd.DataFrame(
            [(k, getattr(cfg, k)) for k in cfg.__dict__.keys()],
            columns=["parameter", "value"],
        )

        with pd.ExcelWriter(cfg.report_path, engine="xlsxwriter") as writer:
            cfg_df.to_excel(writer, index=False, sheet_name="parameters")
            df.to_excel(writer, index=False, sheet_name="events")
        return cfg.report_path


# ---------------------------
# Cadence logic
# ---------------------------

def compute_interval_s(activity_date: dt.date, today: dt.date, cfg: Config) -> int:
    """
    - If within [today - window_before_days, today + window_after_days] -> high_interval_s
    - Otherwise scale linearly up to max_slow_interval_s based on day distance.
    """
    start = today - dt.timedelta(days=cfg.window_before_days)
    end = today + dt.timedelta(days=cfg.window_after_days)
    if start <= activity_date <= end:
        return cfg.high_interval_s

    # Distance in full days from the nearest edge of the window.
    if activity_date < start:
        d = (start - activity_date).days
    else:
        d = (activity_date - end).days

    # Scale: day distance 1 -> closer to high; large distance -> max slow
    # Use a simple saturating curve.
    # For d >= 3, we just use max slow.
    if d <= 0:
        return cfg.high_interval_s
    if d >= 3:
        return cfg.max_slow_interval_s

    # Linear interpolation between high and max_slow across d in [1, 2]
    frac = d / 2.0
    interval = cfg.high_interval_s + frac * (cfg.max_slow_interval_s - cfg.high_interval_s)
    return int(round(interval))


# ---------------------------
# Utilities
# ---------------------------

def local_now() -> dt.datetime:
    return dt.datetime.now()

def today_local() -> dt.date:
    return local_now().date()

def jitter(seconds: float, pct: float = 0.1) -> float:
    """Add ±pct jitter."""
    delta = seconds * pct
    return max(0.0, seconds + random.uniform(-delta, delta))


# ---------------------------
# HTTP client wrapper
# ---------------------------

class HttpClient:
    def __init__(self, cfg: Config, limiter: TokenBucket, sem: asyncio.Semaphore) -> None:
        self.cfg = cfg
        self.limiter = limiter
        self.sem = sem
        timeout = aiohttp.ClientTimeout(total=self.cfg.request_timeout_s, connect=self.cfg.connect_timeout_s)
        self.session = aiohttp.ClientSession(timeout=timeout, raise_for_status=False)

    async def close(self) -> None:
        await self.session.close()

    async def get_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Tuple[int, Any]:
        await self.limiter.take()
        async with self.sem:
            async with self.session.get(url, headers=self.cfg.headers(), params=params) as resp:
                status = resp.status
                with contextlib.suppress(Exception):
                    data = await resp.json()
                    return status, data
                text = await resp.text()
                return status, {"_raw": text}

    async def post_json(self, url: str, payload: Dict[str, Any]) -> Tuple[int, Any]:
        await self.limiter.take()
        async with self.sem:
            async with self.session.post(url, headers=self.cfg.headers(), json=payload) as resp:
                status = resp.status
                with contextlib.suppress(Exception):
                    data = await resp.json()
                    return status, data
                text = await resp.text()
                return status, {"_raw": text}


# ---------------------------
# Bootstrap & refresh
# ---------------------------

async def fetch_initial_activities(client: HttpClient, cfg: Config) -> List[Activity]:
    """
    Replace this with your real bootstrap logic. Expected JSON shape example:
    [{ "loan_alias": "A123", "activity_date": "2025-08-20" }, ...]
    """
    logger.info("Fetching initial activities...")
    params = {"page_size": cfg.activities_page_size}
    if cfg.activities_bootstrap_method.upper() == "POST":
        status, data = await client.post_json(cfg.activities_bootstrap_url, payload=params)
    else:
        status, data = await client.get_json(cfg.activities_bootstrap_url, params=params)

    if status >= 400:
        raise RuntimeError(f"Bootstrap failed: HTTP {status}: {data}")

    items = []
    for row in data or []:
        try:
            ad = dt.date.fromisoformat(row["activity_date"])
            items.append(Activity(loan_alias=row["loan_alias"], activity_date=ad))
        except Exception as e:
            logger.warning("Skipping bad row %s (%s)", row, e)

    logger.info("Bootstrapped %d activities.", len(items))
    return items


async def refresh_activities_daily(run_event: asyncio.Event,
                                   activities_ref: Dict[str, Activity],
                                   client: HttpClient,
                                   cfg: Config) -> None:
    """
    Refresh the global activities list once per day at configured local hour.
    """
    logger.info("Starting daily refresh task.")
    while not run_event.is_set():
        now = local_now()
        tgt = now.replace(hour=cfg.activities_refresh_hour_local, minute=0, second=0, microsecond=0)
        if tgt <= now:
            tgt = tgt + dt.timedelta(days=1)
        wait = (tgt - now).total_seconds()
        try:
            await asyncio.wait_for(run_event.wait(), timeout=wait)
            break
        except asyncio.TimeoutError:
            pass  # time to refresh

        try:
            new_list = await fetch_initial_activities(client, cfg)
            activities_ref.clear()
            activities_ref.update({a.loan_alias: a for a in new_list})
            logger.info("Activities refreshed: %d items.", len(activities_ref))
        except Exception as e:
            logger.exception("Daily refresh failed: %s", e)


# ---------------------------
# Worker
# ---------------------------

async def poll_once(activity: Activity, client: HttpClient, cfg: Config) -> Tuple[bool, Optional[bool], Optional[int], Optional[str], Dict[str, Any]]:
    """
    Return: (success, should_promote, http_status, error, response_json)
    """
    # Build poll request
    params = {"loan_alias": activity.loan_alias}
    url = cfg.status_poll_url

    attempt = 0
    last_error = None
    last_status = None
    data: Dict[str, Any] = {}
    while attempt < cfg.retry_attempts:
        attempt += 1
        try:
            status, payload = await client.get_json(url, params=params)
            last_status = status
            data = payload if isinstance(payload, dict) else {"payload": payload}
            if 200 <= status < 300:
                # Determine if should promote
                promote = False
                try:
                    lr = data.get("loanRepricing") or {}
                    promote = lr.get("statusCode") is not None
                except Exception:
                    promote = False

                # Push to DB sync endpoint (POST)
                sync_payload = {
                    "loan_alias": activity.loan_alias,
                    "activity_date": activity.activity_date.isoformat(),
                    "fetched_at": local_now().isoformat(),
                    "status_payload": data,
                }
                s2, _ = await client.post_json(cfg.db_sync_post_url, payload=sync_payload)
                if 200 <= s2 < 300:
                    return True, promote, status, None, data
                else:
                    last_error = f"DB sync HTTP {s2}"
            else:
                last_error = f"HTTP {status}"
        except Exception as e:
            last_error = f"exception: {e}"

        # Backoff + jitter
        delay = min(cfg.retry_max_delay_s, cfg.retry_base_delay_s * (2 ** (attempt - 1)))
        await asyncio.sleep(jitter(delay, 0.25))

    return False, None, last_status, last_error, data


async def worker(activity: Activity,
                 client: HttpClient,
                 cfg: Config,
                 run_event: asyncio.Event,
                 report: ReportCollector) -> None:
    """
    Per-activity forever worker with dynamic cadence and auto-promotion.
    """
    log = logging.getLogger(f"worker[{activity.loan_alias}]")
    promoted = False
    attempt_counter = 0

    while not run_event.is_set():
        today = today_local()
        interval = cfg.high_interval_s if promoted else compute_interval_s(activity.activity_date, today, cfg)

        # Do the poll
        ok, should_promote, http_status, error, _payload = await poll_once(activity, client, cfg)

        attempt_counter += 1
        if report.enabled:
            report.add(ReportItem(
                timestamp=local_now(),
                loan_alias=activity.loan_alias,
                activity_date=activity.activity_date,
                attempt=attempt_counter,
                success=ok,
                http_status=http_status,
                error=error,
                promoted_to_high=bool(should_promote),
                interval_used_s=interval,
            ))

        if ok:
            if should_promote and not promoted:
                promoted = True
                log.info("Auto-promoted to high frequency (statusCode detected).")
        else:
            log.warning("Poll failed: %s (status=%s)", error, http_status)

        # Sleep with jitter; also bail early if shutdown
        sleep_s = jitter(float(interval), 0.1)
        try:
            await asyncio.wait_for(run_event.wait(), timeout=sleep_s)
            break
        except asyncio.TimeoutError:
            pass


# ---------------------------
# Orchestrator
# ---------------------------

async def main_async(cfg: Config) -> int:
    setup_logging(cfg.log_level)
    logger.info("Starting %s", cfg.service_name)

    # Global coordinating event for shutdown
    run_event = asyncio.Event()

    # Graceful shutdown
    def _handle_sigint():
        logger.info("SIGINT received, shutting down...")
        run_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, _handle_sigint)

    # Downstream protections
    limiter = TokenBucket(rate=cfg.rps_limit, capacity=cfg.burst)
    sem = asyncio.Semaphore(cfg.max_concurrency)
    client = HttpClient(cfg, limiter, sem)

    report = ReportCollector(enabled=cfg.report_enabled)

    activities_ref: Dict[str, Activity] = {}

    try:
        # Bootstrap
        initial = await fetch_initial_activities(client, cfg)
        activities_ref.update({a.loan_alias: a for a in initial})

        # Spawn worker tasks
        workers: Dict[str, asyncio.Task] = {}

        def ensure_worker(a: Activity):
            if a.loan_alias in workers and not workers[a.loan_alias].done():
                return
            workers[a.loan_alias] = asyncio.create_task(worker(a, client, cfg, run_event, report))

        for a in activities_ref.values():
            ensure_worker(a)

        # Daily refresher
        refresher = asyncio.create_task(refresh_activities_daily(run_event, activities_ref, client, cfg))

        # Main loop: monitor set changes (e.g., after refresh) & maintain workers
        while not run_event.is_set():
            # Reconcile current workers with activities_ref every minute
            await asyncio.sleep(60)
            # Start new workers for any new activities
            for a in activities_ref.values():
                ensure_worker(a)
            # Optionally cancel workers for removed activities
            removed = set(workers.keys()) - set(activities_ref.keys())
            for alias in removed:
                t = workers.pop(alias)
                t.cancel()

        # Shutdown path
        logger.info("Stopping workers...")
        for t in workers.values():
            t.cancel()
        await asyncio.gather(*workers.values(), return_exceptions=True)

        refresher.cancel()
        with contextlib.suppress(Exception):
            await refresher

    finally:
        await client.close()

    # Finalize report if enabled
    if cfg.report_enabled:
        path = report.write_excel(cfg)
        if path:
            logger.info("Report written to: %s", path)

    logger.info("Shutdown complete.")
    return 0


# ---------------------------
# CLI
# ---------------------------

def parse_cli_overrides(cfg: Config) -> Config:
    import argparse
    p = argparse.ArgumentParser(description="Near-realtime status sync service")
    p.add_argument("--report", action="store_true", help="Enable Excel report on Ctrl+C")
    p.add_argument("--report-path", type=str, default=cfg.report_path)
    p.add_argument("--log-level", type=str, default=cfg.log_level)
    p.add_argument("--high-interval", type=int, default=cfg.high_interval_s)
    p.add_argument("--max-slow-interval", type=int, default=cfg.max_slow_interval_s)
    p.add_argument("--window-before-days", type=int, default=cfg.window_before_days)
    p.add_argument("--window-after-days", type=int, default=cfg.window_after_days)
    p.add_argument("--rps", type=float, default=cfg.rps_limit)
    p.add_argument("--burst", type=int, default=cfg.burst)
    p.add_argument("--max-concurrency", type=int, default=cfg.max_concurrency)
    p.add_argument("--status-url", type=str, default=cfg.status_poll_url)
    p.add_argument("--sync-url", type=str, default=cfg.db_sync_post_url)
    p.add_argument("--bootstrap-url", type=str, default=cfg.activities_bootstrap_url)
    p.add_argument("--bootstrap-method", type=str, choices=["GET", "POST"], default=cfg.activities_bootstrap_method)
    p.add_argument("--refresh-hour", type=int, default=cfg.activities_refresh_hour_local)

    args = p.parse_args()

    cfg.report_enabled = args.report or cfg.report_enabled
    cfg.report_path = args.report_path
    cfg.log_level = args.log_level
    cfg.high_interval_s = args.high_interval
    cfg.max_slow_interval_s = args.max_slow_interval
    cfg.window_before_days = args.window_before_days
    cfg.window_after_days = args.window_after_days
    cfg.rps_limit = args.rps
    cfg.burst = args.burst
    cfg.max_concurrency = args.max_concurrency
    cfg.status_poll_url = args.status_url
    cfg.db_sync_post_url = args.sync_url
    cfg.activities_bootstrap_url = args.bootstrap_url
    cfg.activities_bootstrap_method = args.bootstrap_method
    cfg.activities_refresh_hour_local = args.refresh_hour
    return cfg


def main() -> int:
    cfg = Config()
    cfg = parse_cli_overrides(cfg)
    return asyncio.run(main_async(cfg))


if __name__ == "__main__":
    raise SystemExit(main())

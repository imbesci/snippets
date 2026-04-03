# Linear Mirror — Architecture & Implementation Prompt

> **What this document is:** A complete architectural blueprint and implementation prompt for an AI coding assistant (Claude Code or similar) to build a lightweight, Linear-inspired project management UI powered by a corporate Jira/Stash/Confluence API. Hand this entire document to your coding agent as the initial prompt.

---

## Project Overview

Build a fullstack application called **Linear Mirror** — a fast, keyboard-driven, Linear-style project management interface that reads from and writes to a corporate Jira instance (with Stash and Confluence integration). The app must feel like Linear: instant, clean, opinionated, and keyboard-first. Jira is the source of truth; this app is a better window into it.

### Core Principles

1. **Jira is the database of record.** Every issue, comment, sprint, and status lives in Jira. We sync it locally for speed but never diverge from Jira's state.
2. **Speed is the feature.** The entire point is that Jira's UI is slow. Every interaction in this app must feel < 100ms. This means aggressive local caching, optimistic updates, and a sync layer that decouples the UI from Jira's API latency.
3. **Opinionated over configurable.** Jira is infinitely configurable. We are not. We pick sensible defaults, map Jira's complex model to Linear's simpler one, and hide complexity unless the user explicitly asks for it.
4. **Keyboard-first, mouse-friendly.** Every action should be reachable via keyboard. The command palette (⌘K) is the primary navigation mechanism.

---

## Tech Stack

| Layer | Technology | Rationale |
|---|---|---|
| **Runtime** | Bun | Fast JS/TS runtime, built-in bundler, native SQLite driver |
| **Framework** | Next.js (App Router) | Server components for initial load, client components for interactivity, API routes for Jira proxy |
| **Language** | TypeScript (strict mode) | End-to-end type safety |
| **Database** | SQLite via `bun:sqlite` | Zero-config, single-file persistence, fast reads for local cache |
| **Styling** | Tailwind CSS v4 | Utility-first, matches Linear's clean aesthetic |
| **State Management** | TanStack Query v5 | Optimistic updates, cache invalidation, background refetching |
| **Keyboard/Command Palette** | cmdk (⌘K library) | Linear uses this exact library |
| **Icons** | Lucide React | Clean, consistent icon set matching Linear's aesthetic |
| **HTTP Client** | Built-in fetch (Bun) | No axios needed; Bun's fetch is fast and native |

### Why Bun + SQLite

Bun has a native SQLite driver (`bun:sqlite`) that is synchronous and extremely fast — no ORM overhead, no connection pooling, no external database process. This gives us a zero-dependency persistence layer for caching Jira data locally. The sync service reads from Jira's API and writes to SQLite; the Next.js API routes read from SQLite and serve the frontend. Jira's API latency is completely hidden from the user.

---

## Project Structure

```
linear-mirror/
├── bun.lockb
├── bunfig.toml
├── package.json
├── tsconfig.json
├── next.config.ts
├── tailwind.config.ts
├── .env.local                          # Jira credentials and config
├── db/
│   ├── schema.sql                      # SQLite schema definitions
│   ├── migrations/                     # Incremental schema migrations
│   └── seed.ts                         # Optional: seed with sample data for dev
├── src/
│   ├── lib/
│   │   ├── db.ts                       # SQLite connection singleton (bun:sqlite)
│   │   ├── jira/
│   │   │   ├── client.ts               # Jira REST API client (typed)
│   │   │   ├── types.ts                # Jira API response types
│   │   │   ├── mappers.ts              # Jira → Linear model mappers
│   │   │   └── webhooks.ts             # Webhook payload handlers
│   │   ├── stash/
│   │   │   ├── client.ts               # Stash/Bitbucket Server API client
│   │   │   └── types.ts
│   │   ├── confluence/
│   │   │   ├── client.ts               # Confluence API client
│   │   │   └── types.ts
│   │   ├── sync/
│   │   │   ├── engine.ts               # Core sync orchestrator
│   │   │   ├── reconciler.ts           # Conflict resolution logic
│   │   │   ├── scheduler.ts            # Polling intervals and webhook fallback
│   │   │   └── queue.ts                # Write-back queue (local → Jira)
│   │   └── utils/
│   │       ├── jql.ts                  # JQL query builder
│   │       ├── keyboard.ts             # Keyboard shortcut registry
│   │       └── dates.ts                # Date formatting helpers
│   ├── models/
│   │   ├── issue.ts                    # Issue type + SQLite queries
│   │   ├── project.ts                  # Project/Team type + queries
│   │   ├── cycle.ts                    # Sprint/Cycle type + queries
│   │   ├── label.ts                    # Label type + queries
│   │   ├── user.ts                     # User/Assignee type + queries
│   │   ├── comment.ts                  # Comment type + queries
│   │   ├── activity.ts                 # Activity/changelog type + queries
│   │   └── pull-request.ts             # PR type + queries (from Stash)
│   ├── app/
│   │   ├── layout.tsx                  # Root layout: sidebar + command palette
│   │   ├── page.tsx                    # Home: inbox / my issues
│   │   ├── team/
│   │   │   └── [teamKey]/
│   │   │       ├── page.tsx            # Team issue list (main view)
│   │   │       ├── board/
│   │   │       │   └── page.tsx        # Kanban board view
│   │   │       ├── cycles/
│   │   │       │   └── page.tsx        # Sprint/cycle list
│   │   │       └── backlog/
│   │   │           └── page.tsx        # Backlog view
│   │   ├── issue/
│   │   │   └── [issueKey]/
│   │   │       └── page.tsx            # Issue detail view
│   │   ├── settings/
│   │   │   └── page.tsx                # Connection settings, field mapping
│   │   └── api/
│   │       ├── issues/
│   │       │   └── route.ts            # CRUD proxy → SQLite + write-back queue
│   │       ├── sync/
│   │       │   ├── trigger/
│   │       │   │   └── route.ts        # Manual sync trigger
│   │       │   └── webhook/
│   │       │       └── route.ts        # Jira webhook receiver
│   │       ├── search/
│   │       │   └── route.ts            # Full-text search against SQLite
│   │       └── health/
│   │           └── route.ts            # Sync status + Jira connectivity check
│   ├── components/
│   │   ├── layout/
│   │   │   ├── Sidebar.tsx             # Left nav: teams, views, filters
│   │   │   ├── CommandPalette.tsx       # ⌘K command palette
│   │   │   └── Breadcrumbs.tsx
│   │   ├── issues/
│   │   │   ├── IssueList.tsx           # Virtualized issue list (main table)
│   │   │   ├── IssueRow.tsx            # Single row in issue list
│   │   │   ├── IssueDetail.tsx         # Right panel / full-page detail
│   │   │   ├── IssueCreateModal.tsx    # New issue creation
│   │   │   ├── IssueStatusBadge.tsx    # Status pill with color
│   │   │   ├── IssuePriorityIcon.tsx   # Priority indicator
│   │   │   └── IssueFilters.tsx        # Filter bar
│   │   ├── board/
│   │   │   ├── KanbanBoard.tsx         # Drag-and-drop board
│   │   │   └── KanbanColumn.tsx        # Single status column
│   │   ├── cycles/
│   │   │   ├── CycleList.tsx           # Sprint list
│   │   │   └── CycleProgress.tsx       # Sprint progress bar
│   │   ├── activity/
│   │   │   ├── ActivityFeed.tsx        # Changelog / activity stream
│   │   │   └── CommentThread.tsx       # Comments on an issue
│   │   ├── common/
│   │   │   ├── Avatar.tsx              # User avatar
│   │   │   ├── Badge.tsx               # Generic badge/pill
│   │   │   ├── Tooltip.tsx
│   │   │   ├── ContextMenu.tsx         # Right-click context menus
│   │   │   ├── Dropdown.tsx
│   │   │   └── VirtualList.tsx         # Virtualized scrolling wrapper
│   │   └── pr/
│   │       └── PullRequestLink.tsx     # Stash PR card on issue detail
│   └── hooks/
│       ├── useIssues.ts                # TanStack Query hook for issues
│       ├── useIssue.ts                 # Single issue query + mutations
│       ├── useSearch.ts                # Search hook
│       ├── useKeyboardShortcuts.ts     # Global keyboard handler
│       ├── useOptimisticUpdate.ts      # Generic optimistic update pattern
│       └── useSyncStatus.ts            # Sync health indicator
└── scripts/
    ├── sync-full.ts                    # Full sync script (run on first setup)
    ├── sync-daemon.ts                  # Long-running sync process
    └── migrate.ts                      # Run DB migrations
```

---

## SQLite Schema

This is the local cache schema. It mirrors the subset of Jira's data model that we care about, flattened into Linear's simpler concepts.

```sql
-- db/schema.sql

PRAGMA journal_mode = WAL;          -- Write-ahead logging for concurrent reads
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 5000;

-- ============================================================
-- SYNC METADATA
-- ============================================================

CREATE TABLE IF NOT EXISTS sync_state (
    entity_type     TEXT PRIMARY KEY,           -- 'issues', 'users', 'sprints', etc.
    last_synced_at  TEXT NOT NULL,              -- ISO 8601 timestamp
    last_jira_updated TEXT,                     -- Last Jira updatedDate we've seen
    cursor          TEXT,                        -- Pagination cursor if sync was interrupted
    status          TEXT DEFAULT 'idle'          -- 'idle', 'syncing', 'error'
);

CREATE TABLE IF NOT EXISTS write_queue (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type     TEXT NOT NULL,              -- 'issue', 'comment', etc.
    entity_id       TEXT NOT NULL,              -- Local or Jira ID
    operation       TEXT NOT NULL,              -- 'create', 'update', 'delete', 'transition'
    payload         TEXT NOT NULL,              -- JSON payload to send to Jira
    status          TEXT DEFAULT 'pending',     -- 'pending', 'in_flight', 'failed', 'completed'
    attempts        INTEGER DEFAULT 0,
    last_error      TEXT,
    created_at      TEXT DEFAULT (datetime('now')),
    processed_at    TEXT
);

-- ============================================================
-- CORE ENTITIES
-- ============================================================

CREATE TABLE IF NOT EXISTS users (
    account_id      TEXT PRIMARY KEY,           -- Jira accountId
    display_name    TEXT NOT NULL,
    email           TEXT,
    avatar_url      TEXT,
    active          INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS teams (
    key             TEXT PRIMARY KEY,           -- Jira project key (e.g., 'ENG')
    name            TEXT NOT NULL,              -- Jira project name
    description     TEXT,
    lead_id         TEXT REFERENCES users(account_id),
    avatar_url      TEXT,
    jira_id         TEXT NOT NULL,              -- Jira project ID (numeric)
    updated_at      TEXT
);

CREATE TABLE IF NOT EXISTS cycles (
    id              TEXT PRIMARY KEY,           -- Jira sprint ID
    team_key        TEXT NOT NULL REFERENCES teams(key),
    name            TEXT NOT NULL,
    goal            TEXT,                        -- Sprint goal
    state           TEXT NOT NULL,              -- 'future', 'active', 'closed'
    start_date      TEXT,
    end_date        TEXT,
    completed_at    TEXT,
    jira_board_id   TEXT,
    updated_at      TEXT
);

CREATE TABLE IF NOT EXISTS labels (
    id              TEXT PRIMARY KEY,           -- Composite: team_key + label_name
    team_key        TEXT REFERENCES teams(key),
    name            TEXT NOT NULL,
    color           TEXT,                        -- Hex color for display
    description     TEXT
);

CREATE TABLE IF NOT EXISTS statuses (
    id              TEXT PRIMARY KEY,           -- Jira status ID
    name            TEXT NOT NULL,              -- e.g., 'In Progress'
    category        TEXT NOT NULL,              -- 'backlog', 'unstarted', 'started', 'completed', 'canceled'
    team_key        TEXT REFERENCES teams(key),
    color           TEXT,
    sort_order      INTEGER DEFAULT 0,
    jira_transition_id TEXT                     -- Transition ID needed to move TO this status
);

CREATE TABLE IF NOT EXISTS issues (
    key             TEXT PRIMARY KEY,           -- e.g., 'ENG-1234'
    team_key        TEXT NOT NULL REFERENCES teams(key),
    title           TEXT NOT NULL,              -- summary
    description     TEXT,                       -- Rendered to markdown from Jira's ADF/wiki
    status_id       TEXT REFERENCES statuses(id),
    priority        TEXT,                       -- 'urgent', 'high', 'medium', 'low', 'none'
    assignee_id     TEXT REFERENCES users(account_id),
    creator_id      TEXT REFERENCES users(account_id),
    cycle_id        TEXT REFERENCES cycles(id), -- Sprint
    parent_key      TEXT,                       -- Epic or parent issue key
    issue_type      TEXT,                       -- 'story', 'bug', 'task', 'epic', 'subtask'
    story_points    REAL,
    due_date        TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    resolved_at     TEXT,
    jira_id         TEXT NOT NULL,              -- Jira issue ID (numeric)
    sort_order      REAL DEFAULT 0              -- For manual ordering within views
);

CREATE TABLE IF NOT EXISTS issue_labels (
    issue_key       TEXT REFERENCES issues(key) ON DELETE CASCADE,
    label_id        TEXT REFERENCES labels(id) ON DELETE CASCADE,
    PRIMARY KEY (issue_key, label_id)
);

CREATE TABLE IF NOT EXISTS comments (
    id              TEXT PRIMARY KEY,           -- Jira comment ID
    issue_key       TEXT NOT NULL REFERENCES issues(key) ON DELETE CASCADE,
    author_id       TEXT REFERENCES users(account_id),
    body            TEXT NOT NULL,              -- Markdown
    created_at      TEXT NOT NULL,
    updated_at      TEXT
);

CREATE TABLE IF NOT EXISTS activity_log (
    id              TEXT PRIMARY KEY,           -- Jira changelog ID
    issue_key       TEXT NOT NULL REFERENCES issues(key) ON DELETE CASCADE,
    author_id       TEXT REFERENCES users(account_id),
    field           TEXT NOT NULL,              -- 'status', 'assignee', 'priority', etc.
    from_value      TEXT,
    to_value        TEXT,
    created_at      TEXT NOT NULL
);

-- ============================================================
-- STASH / BITBUCKET INTEGRATION
-- ============================================================

CREATE TABLE IF NOT EXISTS pull_requests (
    id              TEXT PRIMARY KEY,           -- Stash PR ID
    issue_key       TEXT REFERENCES issues(key),
    repo_slug       TEXT NOT NULL,
    title           TEXT NOT NULL,
    state           TEXT NOT NULL,              -- 'OPEN', 'MERGED', 'DECLINED'
    author_id       TEXT,
    source_branch   TEXT,
    target_branch   TEXT,
    url             TEXT,
    created_at      TEXT,
    updated_at      TEXT
);

-- ============================================================
-- CONFLUENCE INTEGRATION
-- ============================================================

CREATE TABLE IF NOT EXISTS linked_docs (
    id              TEXT PRIMARY KEY,           -- Confluence page ID
    issue_key       TEXT REFERENCES issues(key),
    title           TEXT NOT NULL,
    space_key       TEXT,
    url             TEXT NOT NULL,
    excerpt         TEXT,                        -- First ~200 chars
    updated_at      TEXT
);

-- ============================================================
-- INDEXES
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_issues_team ON issues(team_key);
CREATE INDEX IF NOT EXISTS idx_issues_status ON issues(status_id);
CREATE INDEX IF NOT EXISTS idx_issues_assignee ON issues(assignee_id);
CREATE INDEX IF NOT EXISTS idx_issues_cycle ON issues(cycle_id);
CREATE INDEX IF NOT EXISTS idx_issues_parent ON issues(parent_key);
CREATE INDEX IF NOT EXISTS idx_issues_updated ON issues(updated_at);
CREATE INDEX IF NOT EXISTS idx_issues_type ON issues(issue_type);
CREATE INDEX IF NOT EXISTS idx_comments_issue ON comments(issue_key);
CREATE INDEX IF NOT EXISTS idx_activity_issue ON activity_log(issue_key);
CREATE INDEX IF NOT EXISTS idx_prs_issue ON pull_requests(issue_key);
CREATE INDEX IF NOT EXISTS idx_linked_docs_issue ON linked_docs(issue_key);
CREATE INDEX IF NOT EXISTS idx_write_queue_status ON write_queue(status);

-- Full-text search
CREATE VIRTUAL TABLE IF NOT EXISTS issues_fts USING fts5(
    key,
    title,
    description,
    content=issues,
    content_rowid=rowid
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS issues_ai AFTER INSERT ON issues BEGIN
    INSERT INTO issues_fts(rowid, key, title, description)
    VALUES (new.rowid, new.key, new.title, new.description);
END;

CREATE TRIGGER IF NOT EXISTS issues_ad AFTER DELETE ON issues BEGIN
    INSERT INTO issues_fts(issues_fts, rowid, key, title, description)
    VALUES ('delete', old.rowid, old.key, old.title, old.description);
END;

CREATE TRIGGER IF NOT EXISTS issues_au AFTER UPDATE ON issues BEGIN
    INSERT INTO issues_fts(issues_fts, rowid, key, title, description)
    VALUES ('delete', old.rowid, old.key, old.title, old.description);
    INSERT INTO issues_fts(rowid, key, title, description)
    VALUES (new.rowid, new.key, new.title, new.description);
END;
```

---

## Environment Configuration

```bash
# .env.local

# Jira Configuration
JIRA_BASE_URL=https://your-company.atlassian.net   # Or on-prem URL
JIRA_USERNAME=your.email@company.com
JIRA_API_TOKEN=your-api-token                       # For Cloud; use PAT for Server/DC
JIRA_AUTH_TYPE=basic                                 # 'basic' (cloud), 'bearer' (PAT), 'oauth2'

# Stash / Bitbucket Server
STASH_BASE_URL=https://stash.your-company.com
STASH_USERNAME=your.username
STASH_TOKEN=your-personal-access-token

# Confluence
CONFLUENCE_BASE_URL=https://your-company.atlassian.net/wiki
CONFLUENCE_USERNAME=your.email@company.com
CONFLUENCE_TOKEN=your-api-token

# Sync Configuration
SYNC_POLL_INTERVAL_MS=30000                          # 30 seconds between polls
SYNC_PROJECTS=ENG,PLATFORM,INFRA                     # Comma-separated project keys to sync
SYNC_MAX_ISSUES_PER_POLL=100                         # Page size for incremental sync

# App
SQLITE_DB_PATH=./data/linear-mirror.db
```

---

## Sync Engine Design

The sync engine is the most critical piece of infrastructure. It must be reliable, incremental, and fast.

### Sync Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    SYNC ENGINE                          │
│                                                         │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Webhook  │───▶│  Reconciler  │───▶│   SQLite     │  │
│  │  Receiver │    │              │    │   Writer     │  │
│  └──────────┘    │  - Dedupe    │    └──────────────┘  │
│                  │  - Conflict  │                       │
│  ┌──────────┐    │    resolve   │    ┌──────────────┐  │
│  │  Poller   │───▶│  - Map to   │───▶│   Event      │  │
│  │ (fallback)│    │    schema   │    │   Emitter    │  │
│  └──────────┘    └──────────────┘    └──────┬───────┘  │
│                                             │          │
│  ┌──────────┐                               │          │
│  │  Write    │◀──────────────────────────────┘          │
│  │  Queue    │──────▶ Jira API (create/update/delete)  │
│  └──────────┘                                          │
└─────────────────────────────────────────────────────────┘
```

### Sync Strategy

#### Initial Full Sync (first run)

```
1. For each project in SYNC_PROJECTS:
   a. Fetch all statuses, issue types, and workflow transitions → populate `statuses` table
   b. Fetch all sprints (active + closed recent + future) → populate `cycles` table
   c. Fetch all issues via JQL: `project = {KEY} ORDER BY updated DESC`
      - Paginate through ALL issues (may be thousands)
      - For each issue: map fields → insert into `issues` table
      - Extract labels, comments, changelog → insert into related tables
   d. Fetch linked PRs from Stash for each issue (batch by branch naming convention)
   e. Fetch linked Confluence pages via issue remote links
2. Record sync_state for each entity type
```

#### Incremental Sync (ongoing)

```
1. Check sync_state.last_jira_updated for 'issues'
2. Query Jira: `project IN ({PROJECTS}) AND updated > "{last_updated}" ORDER BY updated ASC`
3. For each returned issue:
   a. If exists locally: update fields, merge comments/changelog
   b. If new: insert with full field mapping
   c. If deleted (check via separate call if suspected): mark as removed
4. Update sync_state.last_jira_updated
5. Process write_queue: send pending local changes to Jira
```

#### Webhook Handler (preferred path when available)

```
POST /api/sync/webhook
1. Validate webhook signature
2. Parse event type: issue_created, issue_updated, sprint_started, etc.
3. Fetch fresh issue data from Jira API (webhooks don't include all fields)
4. Run through reconciler → update SQLite
5. Emit event for any connected clients (future: SSE/WebSocket)
```

### Conflict Resolution Rules

When a local optimistic update conflicts with an incoming Jira change:

1. **Jira wins** for: status transitions, assignee changes, sprint membership (these may have workflow rules)
2. **Last-write-wins** for: title, description, priority, labels (with user notification)
3. **Merge** for: comments (append both), changelog (append both)
4. **Queue retry** for: failed writes (retry 3x with exponential backoff, then surface error to user)

### Write-Back Queue

All mutations go through the write queue, never directly to Jira:

```typescript
// Pseudocode for write flow
async function updateIssue(key: string, changes: Partial<Issue>) {
  // 1. Optimistic local update
  db.prepare("UPDATE issues SET ... WHERE key = ?").run(changes, key);

  // 2. Enqueue Jira write
  db.prepare(`
    INSERT INTO write_queue (entity_type, entity_id, operation, payload)
    VALUES ('issue', ?, 'update', ?)
  `).run(key, JSON.stringify(changes));

  // 3. Process queue (async, non-blocking)
  processWriteQueue(); // fires and forgets
}
```

---

## Jira → Linear Data Model Mapping

This is the conceptual mapping. The `mappers.ts` file should implement these transformations.

| Linear Concept | Jira Source | Mapping Notes |
|---|---|---|
| **Team** | Project | 1:1. Project key becomes team identifier. |
| **Issue** | Issue | 1:1. Key is preserved (e.g., ENG-1234). |
| **Cycle** | Sprint | 1:1. Active sprint = current cycle. |
| **Project** | Epic | Epics become "Projects" in the Linear sense — a grouping above issues. |
| **Sub-issue** | Sub-task | Direct mapping. Parent key stored on child. |
| **Status** | Status | Map to Linear's categories: Backlog, Todo, In Progress, Done, Canceled. Use Jira's status category field. |
| **Priority** | Priority | Map Jira's 1-5 to: Urgent, High, Medium, Low, No Priority. |
| **Label** | Label + Component | Merge Jira labels and components into a single label concept. |
| **Assignee** | Assignee | 1:1. |
| **Comment** | Comment | 1:1. Convert Jira's ADF (Atlassian Document Format) or wiki markup to Markdown. |
| **Activity** | Changelog | 1:1. Each changelog entry becomes an activity item. |
| **PR Link** | Stash PR | Linked via branch name convention or Jira dev panel API. |
| **Doc Link** | Confluence Page | Via issue remote links or applinks. |

### Priority Mapping

```typescript
const PRIORITY_MAP: Record<string, string> = {
  '1': 'urgent',    // Jira: Highest
  '2': 'high',      // Jira: High
  '3': 'medium',    // Jira: Medium
  '4': 'low',       // Jira: Low
  '5': 'none',      // Jira: Lowest
};
```

### Status Category Mapping

```typescript
// Jira status categories → Linear-style grouping
const STATUS_CATEGORY_MAP: Record<string, string> = {
  'new':            'backlog',      // Jira: To Do (unmapped/new)
  'undefined':      'backlog',
  'to do':          'unstarted',
  'in progress':    'started',
  'done':           'completed',
  'complete':       'completed',
};
```

### ADF to Markdown Conversion

Jira Cloud uses Atlassian Document Format (ADF) for rich text. Jira Server uses wiki markup. You need to handle both:

```typescript
// Use a library like 'adf-to-md' or build a lightweight converter
// Key conversions:
// - ADF paragraph → markdown paragraph
// - ADF heading → # markdown heading
// - ADF bulletList → - markdown list
// - ADF codeBlock → ```code block```
// - ADF mention → @displayName
// - ADF inlineCard (Jira link) → [KEY](url)
// - Wiki {code} → ```code```
// - Wiki h1. → # heading
```

---

## API Routes Design

All frontend data fetching goes through Next.js API routes, which read from SQLite (fast) and queue writes to Jira (async).

### Route Map

```
GET    /api/issues?team=ENG&status=started&assignee=me&cycle=active&q=search+term
GET    /api/issues/[key]
POST   /api/issues                          → Create issue (optimistic + queue)
PATCH  /api/issues/[key]                    → Update issue (optimistic + queue)
DELETE /api/issues/[key]                    → Delete issue (soft delete + queue)

POST   /api/issues/[key]/transition         → Status change (needs transition ID)
POST   /api/issues/[key]/comments           → Add comment
GET    /api/issues/[key]/activity           → Changelog + comments merged by time

GET    /api/teams                           → List synced projects
GET    /api/teams/[key]/cycles              → Sprints for a project
GET    /api/teams/[key]/members             → Team members
GET    /api/teams/[key]/labels              → Labels + components

GET    /api/search?q=term                   → Full-text search via FTS5

POST   /api/sync/trigger                    → Force immediate sync
POST   /api/sync/webhook                    → Jira webhook receiver
GET    /api/health                          → Sync status, last sync time, queue depth
```

### Response Shape

All API responses follow a consistent shape:

```typescript
interface ApiResponse<T> {
  data: T;
  meta: {
    total?: number;
    sync: {
      lastSyncedAt: string;
      isStale: boolean;        // true if last sync > 2 minutes ago
    };
  };
}
```

---

## Frontend Architecture

### Layout Structure

```
┌──────────────────────────────────────────────────────────┐
│  ⌘K Command Palette (overlay, always available)          │
├────────────┬─────────────────────────────────────────────┤
│            │                                             │
│  Sidebar   │         Main Content Area                   │
│            │                                             │
│  - Inbox   │  ┌─────────────────────────────────────┐   │
│  - My Work │  │  Filter Bar                          │   │
│  ─────────│  │  [Status ▾] [Assignee ▾] [Priority ▾]│   │
│  Teams:    │  └─────────────────────────────────────┘   │
│  > ENG     │  ┌─────────────────────────────────────┐   │
│    Issues  │  │  Issue List (virtualized)             │   │
│    Board   │  │  ┌─────────────────────────────────┐ │   │
│    Cycles  │  │  │ ENG-123  Fix auth bug    ● High │ │   │
│    Backlog │  │  │ ENG-124  Add dark mode   ○ Med  │ │   │
│  > PLAT    │  │  │ ENG-125  Update deps     ○ Low  │ │   │
│  ─────────│  │  └─────────────────────────────────┘ │   │
│  Settings  │  └─────────────────────────────────────┘   │
│            │                                             │
└────────────┴─────────────────────────────────────────────┘
```

### Key UI Behaviors

1. **Virtualized Lists**: Use `@tanstack/react-virtual` for issue lists. A project may have 10,000+ issues; we must not render them all.

2. **Optimistic Updates**: Every mutation (status change, assign, priority) updates the UI immediately. The write queue handles Jira sync in the background. If it fails, show a subtle toast — don't revert unless the user asks.

3. **Keyboard Navigation**:
   - `j/k` — navigate up/down in issue list
   - `Enter` — open selected issue
   - `Escape` — go back / close panel
   - `⌘K` — open command palette
   - `c` — create new issue
   - `s` — change status (opens status picker)
   - `a` — change assignee
   - `p` — change priority
   - `l` — toggle label
   - `⌘⇧P` — switch between list/board view

4. **Command Palette (⌘K)**:
   - Search issues across all teams
   - Navigate to any team/view
   - Run actions: "Assign ENG-123 to me", "Move to In Progress"
   - Toggle settings

5. **Issue Detail Panel**: Opens as a side panel (not a new page) by default. Can be expanded to full page. Shows: title (editable inline), description (editable, markdown), status/priority/assignee (click to change), activity feed (comments + changelog interleaved by time), linked PRs (from Stash), linked docs (from Confluence).

6. **Board View**: Kanban columns derived from status categories. Drag-and-drop triggers a status transition (which goes through the write queue → Jira transition API).

7. **Filters**: Filters are composable and persist in the URL (query params). Support: status, assignee, priority, label, cycle, issue type, text search. Filters operate against SQLite, not Jira JQL.

### Styling Direction

Match Linear's aesthetic precisely:

- **Background**: Very subtle warm gray (#1C1C1E for dark mode, or very light gray for light mode)
- **Font**: System font stack (Linear uses Inter — we can use it or a similar sans-serif)
- **Spacing**: Generous but consistent. 8px grid system.
- **Borders**: 1px, very low contrast (border-gray-200 / border-gray-800)
- **Accents**: Purple for primary actions (Linear's signature), but keep it understated
- **Icons**: Lucide, 16px, muted color unless active
- **Animations**: Subtle. 150ms ease transitions on hover states. No bouncing or spring animations.
- **Dark mode first**: Linear is dark-mode-first. Default to dark, support light.
- **No rounded corners on cards**: Linear uses very subtle rounding (4px max). Avoid pill shapes.

---

## Jira API Client

### Authentication Patterns

```typescript
// src/lib/jira/client.ts

type AuthConfig =
  | { type: 'basic'; username: string; token: string }   // Cloud
  | { type: 'bearer'; token: string }                     // Server/DC PAT
  | { type: 'oauth2'; accessToken: string };              // OAuth 2.0

function buildHeaders(auth: AuthConfig): Record<string, string> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  };

  switch (auth.type) {
    case 'basic':
      headers['Authorization'] = `Basic ${btoa(`${auth.username}:${auth.token}`)}`;
      break;
    case 'bearer':
      headers['Authorization'] = `Bearer ${auth.token}`;
      break;
    case 'oauth2':
      headers['Authorization'] = `Bearer ${auth.accessToken}`;
      break;
  }

  return headers;
}
```

### Key Jira REST Endpoints

```typescript
// Issues
GET  /rest/api/3/search?jql={jql}&maxResults=100&startAt=0&expand=changelog,renderedFields
GET  /rest/api/3/issue/{issueKey}?expand=changelog,renderedFields
POST /rest/api/3/issue
PUT  /rest/api/3/issue/{issueKey}
POST /rest/api/3/issue/{issueKey}/transitions   // Status changes

// Comments
GET  /rest/api/3/issue/{issueKey}/comment
POST /rest/api/3/issue/{issueKey}/comment

// Sprints (via Agile API)
GET  /rest/agile/1.0/board/{boardId}/sprint?state=active,future,closed
GET  /rest/agile/1.0/sprint/{sprintId}/issue

// Projects
GET  /rest/api/3/project
GET  /rest/api/3/project/{projectKey}

// Users
GET  /rest/api/3/user/search?query={query}

// Webhooks (register)
POST /rest/api/3/webhook

// Note: Jira Server/DC uses /rest/api/2/ instead of /rest/api/3/
// The client should abstract this version difference
```

### Rate Limiting

Jira Cloud has rate limits (~100 requests/minute for basic auth). The sync engine must:

1. Track remaining rate limit via response headers (`X-RateLimit-Remaining`)
2. Implement exponential backoff on 429 responses
3. Batch requests where possible (JQL search returns 100 results per page)
4. Prioritize write-queue processing over polling reads

---

## Detailed Implementation Order

Build in this sequence. Each phase should be deployable and usable.

### Phase 1: Foundation (Day 1-2)

```
1. Initialize project: bun create next-app with TypeScript + Tailwind
2. Set up SQLite: create db.ts singleton, run schema.sql
3. Build Jira client: auth, basic GET/POST, error handling, rate limiting
4. Build initial sync: full sync script that populates all tables
5. Verify: run sync, inspect SQLite with `bun run scripts/sync-full.ts`
```

### Phase 2: Read-Only UI (Day 3-5)

```
1. Build layout: Sidebar + main content area
2. Build IssueList with virtualization
3. Build IssueDetail panel (side panel view)
4. Build filter bar (status, assignee, priority)
5. Add team switching in sidebar
6. Wire up TanStack Query hooks to API routes
7. Add ⌘K command palette with navigation commands
8. Add keyboard navigation (j/k/Enter/Escape)
```

### Phase 3: Write Operations (Day 6-7)

```
1. Implement write queue processor
2. Add status change (transition API)
3. Add assignee change
4. Add priority change
5. Add inline title editing
6. Add comment creation
7. Add issue creation modal
8. Wire up optimistic updates for all mutations
```

### Phase 4: Board View + Cycles (Day 8-9)

```
1. Build KanbanBoard with drag-and-drop
2. Wire drag-drop to status transitions
3. Build CycleList and CycleProgress
4. Add cycle filtering to issue views
5. Add backlog view (issues not in any cycle)
```

### Phase 5: Integrations + Polish (Day 10-12)

```
1. Build Stash PR integration on issue detail
2. Build Confluence doc links on issue detail
3. Add sync daemon (continuous background sync)
4. Add webhook receiver (if Jira webhooks are available)
5. Add sync health indicator in sidebar
6. Add error handling and retry UI for failed writes
7. Add dark/light mode toggle
8. Performance tuning: ensure all views render < 100ms
```

### Phase 6: Advanced Features (Stretch)

```
1. Bulk actions (multi-select + batch status/assignee change)
2. Custom views (saved filter combinations)
3. Notifications (poll Jira notifications API)
4. Graph/progress views (burndown from sprint data)
5. SSE/WebSocket for real-time updates between browser tabs
```

---

## Critical Implementation Notes

### 1. Handling Jira's Field Sprawl

Every Jira instance has custom fields. The app should:

- Start with standard fields only (summary, description, status, priority, assignee, sprint, labels, components, story points)
- Store unknown fields as a JSON blob in a `custom_fields TEXT` column on `issues`
- Provide a settings page where users can map custom field IDs to display names
- Common custom field IDs vary by instance; story points might be `customfield_10016` on one instance and `customfield_10028` on another

### 2. Jira Server vs Cloud Differences

| Concern | Cloud | Server/Data Center |
|---|---|---|
| API version | `/rest/api/3/` | `/rest/api/2/` |
| Auth | Basic (email + API token) | PAT or session cookie |
| Rich text | ADF (JSON) | Wiki markup |
| User IDs | `accountId` (opaque) | `username` |
| Webhooks | Via Connect app or REST | Via admin UI |
| Agile API | Same | Same |

**Build the Jira client with a config-driven adapter layer** so the same sync engine works against both Cloud and Server.

### 3. Pagination

Jira paginates via `startAt` + `maxResults`. The sync engine must handle:
- Large result sets (> 1000 issues) across many pages
- Interruption recovery (store cursor in `sync_state`)
- Changed results between pages (use `ORDER BY updated ASC` and track last seen timestamp)

### 4. Error Boundaries

The UI must never crash due to a sync failure or malformed Jira data. Every component that displays Jira-sourced data should:
- Have a React error boundary
- Show graceful degradation (e.g., "Could not load description" instead of a white screen)
- Surface sync errors in a non-blocking notification area (bottom-right toast stack)

### 5. Security

- **Never expose Jira credentials to the browser.** All Jira API calls go through Next.js API routes (server-side only).
- Store `.env.local` credentials securely. Consider supporting OS keychain integration for local dev.
- The SQLite database file contains all synced Jira data. It should be `.gitignore`d and backed up appropriately.
- If deploying for a team (not just local), add session auth in front of the app.

---

## Testing Strategy

```
1. Unit tests (Bun test runner):
   - Jira → Linear model mappers
   - JQL query builder
   - Conflict resolution logic
   - SQLite query functions

2. Integration tests:
   - Sync engine against recorded Jira API responses (use fixtures)
   - Write queue processing
   - API routes against seeded SQLite

3. E2E tests (Playwright):
   - Issue list loads and virtualizes correctly
   - Keyboard navigation works
   - Status change roundtrips (optimistic → queued → confirmed)
   - Command palette search and navigation
```

---

## Quick Start Commands

```bash
# Install dependencies
bun install

# Set up database
bun run scripts/migrate.ts

# Run initial sync (first time only — may take a few minutes)
bun run scripts/sync-full.ts

# Start dev server
bun run dev

# Start sync daemon (in separate terminal)
bun run scripts/sync-daemon.ts

# Run tests
bun test
```

---

## Summary

This is a sync-layer-first architecture. The entire UX quality depends on the local SQLite cache being fast and fresh. Build the sync engine and data model first, verify it works against your actual Jira instance, then layer the UI on top. The UI itself is deliberately simple — it's a fast list view, a side panel, a board view, and a command palette. The complexity lives in the sync layer and the Jira API adapter.

Do not try to replicate 100% of Jira's functionality. The goal is to replicate 100% of Linear's simplicity using Jira's data. When in doubt, leave a feature out. You can always add it later; you can never un-complicate a UI.

# AllyH: Coherence-First AI Agent Harness — Architecture & Implementation Plan

**Project Name:** AllyH  
**Stack:** Rust (`rig-core`, `rig-lancedb`, `lancedb`, `arrow-array`, `ratatui`, `tokio`, `tree-sitter`, `git2`, `notify`)  
**Interface:** Interactive Terminal User Interface (TUI)  
**Model Strategy:** Hybrid Routing (Frontier Cloud APIs + Local Quantized Models)  
**Storage Backend:** LanceDB (Local Disk & S3/Object-Store Shared Multi-Agent Backend)

---

## 1. Vision & Core Problem Statement

Modern LLMs possess sufficient raw coding intelligence, yet fail in long-running software engineering tasks due to **long-term coherence decay**. Users are forced to manually refresh context by littering repositories with static Markdown plans (`PLAN.md`, `TODO.md`).

**AllyH** acts as a lightweight, out-of-tree operating system around `rig-core`. Powered by **LanceDB**, it maintains native cross-session episodic memory, hybrid vector + full-text code retrieval, live environment awareness, prescriptive compiler feedback, and explicit interactive control gates—without polluting the target workspace, while remaining ready to scale to shared object storage (`s3://` / MinIO / GCS) across multiple nodes or subagents.

---

## 2. Research Foundations & Design Principles

| Research Pillar | Core Insight | AllyH Architectural Translation |
| :--- | :--- | :--- |
| **Long-Running Harnesses** (Anthropic) | One-shotting exhausts context; Markdown plans are easily corrupted by LLMs. | Enforce single-feature incremental turns backed by a strict structured state machine (`tasks`) stored out-of-tree in LanceDB, plus mandatory startup environment checks. |
| **Interactive Multi-Turn Reality** (*SWE-Interact*) | Frontier models lose ~23–24 percentage points when requirements evolve interactively across turns. | Enforce explicit **Discovery & Plan Approval Gates** in the AllyH TUI before granting autonomous write/execution permissions. |
| **Bi-Temporal & Conflict-Aware Memory** (*Zep/Graphiti*, *Mem0*, *MemoryAgentBench*) | Unbounded vector RAG accumulates stale facts and reinforces past mistakes. | Implement logical invalidation (`valid_at` / `invalid_at`) and pre-write conflict resolution (deduplicate, update, or invalidate) via LanceDB `merge_insert` / `update` operations gated by AllyH, not the LLM. |
| **Sleep-Time Compute** (UC Berkeley / Letta) | Idle-time offline reasoning amortizes context preparation cost by ~2.5x–5x, provided edit permissions are locked. | Run an asynchronous local background worker during user idle time to distill session traces, compact Lance fragments, and update architectural invariants with **read-only** repository access. |
| **Agent-Computer Interface & Compiler Diagnostics** (*SWE-agent*, *CoDAIM 2026*) | Actionable, prescriptive diagnostics improve repair accuracy by +22%+ even on 7B models. | Group file operations into compact tools, verify edits via file hashes, and automatically append structured `cargo check` / `clippy` diagnostics to every `ToolOutput`. |
| **Code Dependency Graphs** (*RepoGraph*) | Reading entire files saturates context windows rapidly. | Use `tree-sitter` to build a symbol/reference graph stored in LanceDB and inject compact 1-hop/2-hop *ego-graphs* around target symbols into the dynamic preamble. |
| **Adaptive Harness & Hierarchical Routing** (*MemoHarness*, *TO-Router*) | Different tasks require different context budgets and model tiers. | Route heavy reasoning/coding to frontier APIs (or large local models) while delegating context compression, diff summarization, and embeddings to fast local models. |

---

## 3. System Architecture & Concurrency Model

The `allyh` binary runs an event-driven Tokio actor topology so the 60 FPS TUI never blocks on LLM inference, LanceDB Arrow batch writes, or compiler checks.

```text
┌──────────────────────────────────────────────────────────────────────┐
│                      AllyH TUI Loop (Ratatui)                        │
│  [Left 70%: Streamed Chat & Tool Cards] │ [Right 30%: Live Memory]   │
└───────────────▲──────────────────────────────────┬───────────────────┘
                │ UiEvent (Tokens/State/Gates)     │ UserCommand
┌───────────────┴──────────────────────────────────▼───────────────────┐
│                 AllyH Core Orchestrator (rig-core)                   │
│  1. Context Assembler (Dynamic Preamble + RepoGraph Ego-Graph)       │
│  2. Control Gate (Discovery -> Plan Approval -> Execution)           │
│  3. Hybrid Router (Frontier API vs. Local OpenAI-Compatible Server)  │
└───▲───────────────────────────┬──────────────────────────────────┬───┘
    │ EnvDelta                  │ TurnCompleted                    │
┌───┴────────────────────┐  ┌───▼──────────────────────────────────▼───┐
│  Environment Watcher   │  │     Memory Consolidator (Async/Idle)     │
│  - notify (FS changes) │  │  - Local LLM (e.g., 9B–12B quantized)    │
│  - git2 (HEAD/diffs)   │  │  - Bi-Temporal Merge & Conflict Gate     │
│  - cargo check hooks   │  │  - LanceDB (Local Dir or S3/MinIO URI)   │
└────────────────────────┘  └──────────────────────────────────────────┘
```

---

## 4. Workspace Crate Structure

```text
allyh/
├── Cargo.toml                  # Workspace definition
└── crates/
    ├── allyh-core/             # State machine, Rig loop, dynamic preamble assembler, gates
    ├── allyh-memory/           # LanceDB connection manager, Arrow schemas, hybrid search, sleep-time worker
    ├── allyh-env/              # notify FS watcher, git2 tracker, tree-sitter RepoGraph, cargo runner
    ├── allyh-tools/            # Compact ACI rig::tool::Tool implementations with post-edit hooks
    └── allyh-tui/              # Ratatui + Crossterm binary (`allyh`), multiline input, live inspector
```

---

## 5. Out-of-Tree Shared Storage Schema (LanceDB / Apache Arrow)

All AllyH state connects via a configurable storage URI:
*   **Local Mode (Default):** `~/.local/share/allyh/db`
*   **Shared / Multi-Node Mode:** `s3://allyh-memory-bucket/shared-db` (MinIO / AWS S3 / GCS)

Every record is partitioned by `repo_id` (SHA-256 of the repository origin URL or canonical root) and `scope_id` (supporting multi-agent isolation or cross-team shared knowledge).

### LanceDB Tables

1.  **`sessions`**:
    *   Columns: `session_id` (Utf8), `repo_id` (Utf8), `started_at` (Timestamp), `ended_at` (Timestamp), `rolling_summary` (Utf8), `token_budget_used` (UInt64).
2.  **`architectural_decisions` (Bi-Temporal Invariants + Vector & FTS Index)**:
    *   Columns: `id` (Utf8), `repo_id` (Utf8), `topic` (Utf8), `rule_statement` (Utf8), `rationale` (Utf8), `affected_paths` (List<Utf8>), `vector` (FixedSizeList<Float32>), `valid_at` (Timestamp), `invalid_at` (Timestamp, nullable), `superseded_by` (Utf8, nullable).
    *   *Pre-write gate:* Before committing a new decision, execute a hybrid search (`vector` + BM25 FTS) filtered by `repo_id = '<id>' AND invalid_at IS NULL`. The local consolidator model decides `ADD`, `UPDATE`, `INVALIDATE`, or `NOOP`, executing atomic `merge_insert` / `update` operations.
3.  **`tasks` (Structured Execution Plan)**:
    *   Columns: `id` (Utf8), `repo_id` (Utf8), `parent_id` (Utf8, nullable), `title` (Utf8), `acceptance_criteria` (List<Utf8>), `status` (`pending` | `approved` | `in_progress` | `verified`), `last_error_trace` (Utf8, nullable), `updated_at` (Timestamp).
    *   *Rule:* Tasks cannot transition to `verified` until automated verification (`cargo test` / `cargo check`) passes.
4.  **`code_graph_snapshots` (RepoGraph + File Integrity)**:
    *   Columns: `repo_id` (Utf8), `file_path` (Utf8), `sha256_hash` (Utf8), `last_read_turn` (UInt64), `symbols` (Utf8 JSON), `dependencies` (List<Utf8>), `vector` (FixedSizeList<Float32>).
    *   Detects out-of-band manual user edits between turns via SHA-256 comparison, triggers incremental `tree-sitter` re-indexing, and injects a concise diff summary into the next turn's preamble.

---

## 6. Interactive TUI Design (`ratatui`)

To eliminate "black-box" context drift, AllyH exposes its internal LanceDB state in real time:

1.  **Main Pane (70% Width):**
    *   Streaming Markdown and syntax-highlighted code blocks (`syntect`).
    *   Collapsible **ACI Tool Cards** displaying compact execution summaries (e.g., `edit_symbol(MemoryStore::consolidate) -> cargo check: PASS (0 warnings)`).
    *   **Interactive Gate Modals:** Explicit `[Approve Plan]` / `[Refine]` prompts before multi-file edits begin (*SWE-Interact* mitigation).
2.  **Coherence Inspector Sidebar (30% Width):**
    *   **Live Environment:** Current Git branch, uncommitted manual edits (`dirty` files), build status, and active storage backend indicator (`LanceDB: Local` or `LanceDB: S3 Shared`).
    *   **Active Task Queue:** Current `in_progress` task and acceptance criteria queried from LanceDB.
    *   **Activated Memories:** Top architectural invariants retrieved via hybrid search for the active prompt (with keybindings to pin or logically invalidate stale rules).
3.  **Status Bar:**
    *   Active Reasoning Model (`API: Claude Sonnet` / `Local: Qwen-35B`), Background Worker status (`Local 9B: Idle / Consolidating`), and context window utilization (`18.4k / 128k tokens`).

---

## 7. Phased Implementation Roadmap

### Phase 1: Foundation & LanceDB Out-of-Tree State (`allyh-memory` + `allyh-env`)
*   Initialize Cargo workspace and configure `lancedb` connection abstraction supporting both local filesystem paths and S3/MinIO object-store URIs.
*   Define Apache Arrow schemas for `sessions`, `architectural_decisions`, `tasks`, and `code_graph_snapshots`, creating scalar indexes on `repo_id` and `invalid_at`, plus IVF-PQ / HNSW vector and FTS indexes.
*   Implement `code_graph_snapshots` hashing + `notify` watcher + `git2` status/diff reader to generate `EnvDelta` structs before every turn.
*   Build a lightweight `tree-sitter` indexer for Rust files to populate symbol definitions and cross-file references (*RepoGraph* ego-graphs).

### Phase 2: Rig Integration & Prescriptive ACI (`allyh-core` + `allyh-tools`)
*   Configure `rig-core` and `rig-lancedb` abstractions supporting cloud APIs and local `llama.cpp` / Ollama endpoints (`http://localhost:9931/v1`).
*   Implement compact ACI tools (`inspect_symbol`, `edit_by_hash`, `run_verification`) where `edit_by_hash` rejects stale reads and automatically runs `cargo check --message-format=json` to attach prescriptive compiler diagnostics to the tool response.
*   Build the **Dynamic Preamble Assembler** combining `EnvDelta`, active `tasks`, top-k active `architectural_decisions` (`invalid_at IS NULL`), and 1-hop `RepoGraph` neighborhoods.

### Phase 3: Hybrid Memory Consolidation & Sleep-Time Worker
*   Implement non-destructive history compaction (preserving intact tool call/response pairs while summarizing older turns via the local 9B–12B model).
*   Implement the write-gated **Bi-Temporal Consolidator**: after each completed task or during idle periods (*sleep-time compute*), extract decisions, resolve conflicts against existing records using LanceDB `merge_insert`, and periodically run `table.optimize()` to compact Lance data fragments without modifying repository files.

### Phase 4: Interactive TUI & Acceptance Testing (`allyh-tui`)
*   Build the `ratatui` split-pane layout, multiline input (`tui-textarea`), token streaming channels, and slash commands (`/model`, `/plan`, `/memories`, `/invalidate`, `/db-status`).
*   Run the **5-Step Isolation & Coherence Acceptance Suite**:
    1. Establish an architectural constraint in Session A (e.g., *"Use `thiserror` in `allyh-core` and `anyhow` only in `allyh-tui`"*).
    2. Close AllyH, reopen in Session B, and request a new module—verify automatic adherence without `.md` files in the repo.
    3. Manually edit a file outside AllyH and verify the harness detects the SHA-256 mismatch and injects the diff summary.
    4. Override the architectural rule in Session C and verify `invalid_at` supersedes the old rule without retrieval conflict.
    5. Switch to a separate repository (or second agent instance connected to the same LanceDB URI) and verify strict `repo_id` partition isolation.

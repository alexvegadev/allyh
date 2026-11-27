# AllyH Roadmap

> **Baseline (v0.2.2):** Out-of-tree LanceDB bi-temporal memory, hybrid vector + BM25 retrieval, Control Gate (`Discovery → Plan Approval → Execution`), reusable procedural Skills, Self-Improvement Memory (SIM) & CRSI (`allyh-sim`), hybrid model routing, and 60 FPS Ratatui TUI — currently with **full language & verification support for Rust only** (`tree-sitter-rust` + `cargo`).

This roadmap defines the highest-leverage capabilities to evolve AllyH from a **Rust-only coherence harness** into a **polyglot, semantically grounded, interoperable engineering OS** without compromising its core rule: *zero repository pollution, strict verification gates, and out-of-tree state*.

---

## Strategic Priorities (Why These Features Matter)

| Priority | Feature Pillar | Why It Is a Force Multiplier for AllyH |
| :--- | :--- | :--- |
| **P0** | **Polyglot Language & Toolchain Engine** | Removes the Rust-only bottleneck (`allyh-env`, `allyh-tools`, `allyh-core`) so AllyH works natively across TypeScript/JS, Python, Go, C/C++, Java, and polyglot monorepos with the same RepoGraph + prescriptive diagnostic loop. |
| **P0** | **LSP Semantic Bridge** | Upgrades RepoGraph from syntactic name matching (`tree-sitter`) to true type-resolved references/call hierarchies and sub-second incremental diagnostics across *any* language that ships a language server. |
| **P1** | **Expanded ACI Tools & Atomic Checkpoints** | Closes day-to-day workflow gaps (fast code search, targeted test runs, policy-gated shell execution, and `git2`/worktree rollback checkpoints per approved plan). |
| **P1** | **Dual-Mode MCP (Client + `allyh mcp serve`)** | Lets AllyH consume external MCP tools under its Control Gate *and* expose AllyH's out-of-tree bi-temporal memory, RepoGraph, Skills, and SIM ledger to external IDEs/agents (VS Code, Cursor, Claude Code, Zed). |
| **P2** | **Provider Expansion, Prompt Caching & Cost-Aware CRSI** | Adds Gemini/OpenRouter/Bedrock/Vertex, exploits Anthropic/OpenAI prefix caching on AllyH's structured preamble, and extends SIM/CRSI to optimize both pass rate and token/latency cost. |
| **P2** | **Headless CI, Multi-Agent Worktrees & TUI Polish** | Activates AllyH's existing S3/shared-storage `(repo_id, scope_id)` partitioning for CI automation and coordinated multi-agent swarms, plus diff review and memory time-travel in the TUI. |

---

## Milestone 1: Pluggable Polyglot Language & Toolchain Engine (`v0.3.0`) — Completed

**Goal:** Eliminate every hardcoded Rust/`.rs`/`cargo` assumption across `allyh-env`, `allyh-tools`, `allyh-core`, `allyh-memory`, and `allyh-tui`, introducing first-class support for **Rust, TypeScript/JavaScript, Python, and Go** (plus extensible custom verifiers for any language).

### 1.1 Generalized RepoGraph & Multi-Language `tree-sitter` Indexing (`allyh-env`, `allyh-memory`)
- [x] **Abstract `LanguageIndexer` trait (`crates/allyh-env/src/lang/mod.rs`):** Replace direct `RustIndexer` calls with a `LanguageRegistry` dispatching by file extension / filename.
- [x] **Broaden `SymbolKind` (`crates/allyh-env/src/graph.rs`):** Add polyglot variants (`Class`, `Interface`, `Method`, `Namespace`, `Variable`, `Component`, `Macro`) while keeping backward-compatible serde deserialization for existing snapshots.
- [x] **Add `language` metadata to `CodeGraphSnapshot` (`crates/allyh-memory/src/schema.rs`):** Persist detected language per file snapshot (with schema migration / default fallback for existing LanceDB tables).
- [x] **Implement Tier-1 `tree-sitter` adapters (`crates/allyh-env/src/lang/`):**
  - [x] Rust (`tree-sitter-rust` — migrate existing `RustIndexer`)
  - [x] TypeScript / TSX / JavaScript (`tree-sitter-typescript`, `tree-sitter-javascript`)
  - [x] Python (`tree-sitter-python`)
  - [x] Go (`tree-sitter-go`)
- [x] **Add feature flags in `Cargo.toml`:** Allow compiling with `--features lang-all` (default) or minimal language subsets to keep build times and binary size predictable.
- [x] **Upgrade file discovery (`crates/allyh-env/src/watcher.rs`):** Replace hardcoded `matches!(ext, "rs" | "toml")` and static `is_ignored_entry` lists with `.gitignore` / `.ignore`-aware walking (`ignore::WalkBuilder`) plus language-aware default ignores (`.venv`, `__pycache__`, `node_modules`, `dist`, `build`, `vendor`, `target`).

### 1.2 Pluggable `ToolchainVerifier` & Unified Diagnostics (`allyh-env`, `allyh-tools`, `allyh-core`)
- [x] **Define `VerificationAdapter` trait (`crates/allyh-env/src/verify/`):** Generalize `CargoRunner` and `CargoVerificationReport` into language-agnostic `VerificationReport`, `DiagnosticItem` (preserving `level`, `code`, `message`, `file`, `line`, `column`, `suggestion`), and `VerificationKind` (`Check`, `Lint`, `Clippy`, `Test`).
- [x] **Implement built-in toolchain auto-detectors & structured diagnostic parsers:**
  - [x] **Rust:** `cargo check` / `cargo clippy` / `cargo test` (`--message-format=json`)
  - [x] **TypeScript / JS:** `tsc --noEmit` (compiler error parser), `eslint --format json` / `biome check --reporter=json`, `vitest` / `jest` / `npm test`
  - [x] **Python:** `pyright --outputjson` / `mypy --output=json`, `ruff check --output-format json`, `pytest` failure parser
  - [x] **Go:** `go vet` diagnostics, `golangci-lint run --out-format json`, `go test -json`
  - [x] **Generic SARIF & Custom Command fallback:** Parse SARIF 2.1.0 or `file:line:col` fallback for C/C++, Java, C#, Zig, etc.
- [x] **Support polyglot monorepos:** Detect multiple toolchains by nearest manifest directory (e.g., `backend/Cargo.toml` + `web/tsconfig.json` + `ml/pyproject.toml` + `svc/go.mod`) and route post-edit checks only to the affected sub-project.
- [x] **Out-of-tree `repo_profiles` table (`crates/allyh-memory`):** Persist per-repository language overrides, custom check/lint/test commands, and ignore rules in LanceDB (no `.allyh.toml` pollution required in the repo), editable via `allyh config` and `/config`.

### 1.3 Polyglot ACI Tools, Preamble, Interlock & CLI (`allyh-tools`, `allyh-core`, `allyh-tui`)
- [x] **Generalize `EditByHashTool` (`crates/allyh-tools/src/edit.rs`):** Rename/alias `cargo_package` → `target_scope` and `cargo_check` → `verification` (with `#[serde(alias = "...")]` for backward compatibility).
- [x] **Generalize `RunVerificationTool` (`crates/allyh-tools/src/verify.rs`):** Accept `kind` (`check` | `lint` | `test` | `clippy` alias), optional `target` (package/subdir/test filter), and `toolchain` override; issue `VerificationProof` from any passing adapter.
- [x] **Update `CompletionInterlock` & `HarnessSpec` (`crates/allyh-core/src/interlock.rs`, `harness.rs`):** Read `verification.success` (with `cargo_check.success` fallback) and alias `auto_cargo_check` → `auto_verify` so existing SIM ledgers and `HarnessSpec` records remain 100% valid.
- [x] **Upgrade `DynamicPreambleAssembler::extract_symbol_queries` (`crates/allyh-core/src/preamble.rs`):** Recognize polyglot paths (`.ts`, `.tsx`, `.js`, `.py`, `.go`, etc.) and member expressions (`Foo.bar`, `pkg/mod.Fn`, `Foo::bar`).
- [x] **Update TUI & CLI labels (`crates/allyh-tui`):** Surface detected language(s) and active toolchain(s) in the Coherence Inspector sidebar; generalize `--no-auto-check` and `/refresh` descriptions.

### Milestone 1 Acceptance Criteria
- [x] Polyglot acceptance test suite covering Rust, TypeScript, Python, and Go fixture repos (including a Rust + TypeScript + Python + Go monorepo): `inspect_symbol` returns accurate 1-hop/2-hop ego-graphs, `edit_by_hash` triggers the matching sub-project verifier, and `run_verification` gates `TaskStatus::Verified` with structured diagnostics.
- [x] Existing Rust acceptance suites (`acceptance_suite.rs`, `sim_acceptance_suite.rs`) pass with zero regressions.

---

## Milestone 2: LSP Semantic Bridge (`v0.4.0`)

**Goal:** Layer optional Language Server Protocol (LSP) integration over `tree-sitter` so AllyH gets compiler-accurate cross-file graphs and sub-second diagnostics for *any* language with an LSP server.

### 2.1 LSP Client Lifecycle Manager (`allyh-env`)
- [ ] **Implement stdio JSON-RPC LSP client (`crates/allyh-env/src/lsp/client.rs`):** Handle `initialize`, `initialized`, `textDocument/didOpen`, `didChange`, `didSave`, `didClose`, and clean `shutdown`/`exit`.
- [ ] **Auto-discover & manage per-language servers (`crates/allyh-env/src/lsp/registry.rs`):** Built-in presets for `rust-analyzer`, `typescript-language-server` / `vtsls`, `pyright-langserver` / `basedpyright`, `gopls`, and `clangd`, with lazy startup, crash recovery, and idle shutdown.
- [ ] **Graceful fallback:** Never require an LSP binary to be installed; fall back transparently to `tree-sitter` + CLI `VerificationAdapter` when unavailable.

### 2.2 Semantic RepoGraph & Instant Diagnostics (`allyh-env`, `allyh-tools`)
- [ ] **Hybrid `tree-sitter` + LSP RepoGraph (`crates/allyh-env/src/graph.rs`):** Enrich `EgoGraph` with semantic edges (`textDocument/references`, `textDocument/definition`, `textDocument/implementation`, `callHierarchy/incomingCalls`, `callHierarchy/outgoingCalls`) when an LSP is active, eliminating name-collision noise.
- [ ] **Fast post-edit diagnostics (`crates/allyh-tools/src/edit.rs`):** Pull `textDocument/publishDiagnostics` / `textDocument/diagnostic` right after `edit_by_hash` for near-instant prescriptive feedback, reserving full build/test invocations for `run_verification`.
- [ ] **Expose LSP status in TUI & CLI:** Show active LSP servers (`LSP: rust-analyzer, vtsls`) in the Coherence Inspector and add `/lsp status|restart` commands.

### Milestone 2 Acceptance Criteria
- [ ] Integration tests using a mock stdio LSP server (plus optional live `rust-analyzer` / `pyright` smoke tests) verify semantic ego-graph disambiguation across identically named symbols in separate modules and sub-second diagnostic attachment after `edit_by_hash`.

---

## Milestone 3: Expanded ACI Tools, Code Search & Atomic Checkpoints (`v0.5.0`)

**Goal:** Give the agent the missing read/search/test/shell primitives needed for complex real-world repositories while strengthening AllyH's safety and rollback guarantees.

### 3.1 High-Signal Search & Navigation Tools (`allyh-tools`, `allyh-memory`)
- [ ] **Add `search_code` ACI tool:** Combine fast ripgrep-style regex/literal search (respecting `.gitignore`, capped output budgets) with hybrid semantic search over LanceDB `code_graph_snapshots` (`vector` + BM25) so the agent can find symbols, error strings, and configs in one call.
- [ ] **Add `git_inspect` read-only tool:** Structured access to `git log`, `git diff`, `git blame` (for a file/line range), and branch status without needing a raw shell.
- [ ] **Enhance `edit_by_hash` with multi-hunk atomic edits:** Support batching multiple validated `(target_snippet, replacement_snippet)` hunks in a single file edit (all-or-nothing rollback on mismatch) to cut tool round-trips.

### 3.2 Policy-Gated Shell & Targeted Test Execution (`allyh-tools`, `allyh-core`, `allyh-tui`)
- [ ] **Add `run_command` ACI tool with Command Approval Gate:**
  - [ ] Read-only / allowlisted commands (configurable per `repo_profile`) run automatically with strict timeout and `HarnessSpec` head/tail output truncation.
  - [ ] Mutating or network-touching commands pause for explicit TUI approval (`[Allow Once]` / `[Allow Pattern]` / `[Deny]`), matching the Plan Approval Gate UX.
- [ ] **Targeted test execution in `run_verification`:** Support running a single test name, file, or module filter across `cargo test`, `pytest`, `vitest`/`jest`, and `go test`.

### 3.3 Atomic Plan Checkpoints & Instant Rollback (`allyh-env`, `allyh-core`, `allyh-tui`)
- [ ] **Out-of-tree snapshot checkpoints via `git2`:** Automatically capture a lightweight pre-execution checkpoint (tree/stash ref or out-of-tree patch bundle) whenever a plan transitions `PlanApproval → Execution`.
- [ ] **Interactive Diff & Rollback (`/diff`, `/rollback`):** View all changes made during the current task/plan with `syntect` highlighting and restore cleanly to the pre-plan checkpoint in one command.

### Milestone 3 Acceptance Criteria
- [ ] Unit + integration tests verifying `search_code` hybrid ranking, atomic multi-hunk rollback on partial mismatch, Command Approval Gate refusal when unapproved, and clean file restoration via `/rollback`.

---

## Milestone 4: Dual-Mode MCP (Client & Server) (`v0.6.0`)

**Goal:** Make AllyH both an extensible agent host (MCP Client) and a reusable coherence/memory backend for the wider AI coding ecosystem (MCP Server).

### 4.1 AllyH as an MCP Client (`allyh-tools`, `allyh-core`, `allyh-memory`)
- [ ] **MCP transport & client manager (`crates/allyh-tools/src/mcp_client.rs`):** Support `stdio` and Streamable HTTP / SSE MCP servers, configured out-of-tree in LanceDB (`repo_profiles`) or via CLI flags.
- [ ] **Control Gate enforcement on MCP tools:** Map MCP `ToolAnnotations` (`readOnlyHint`, `destructiveHint`) directly onto AllyH's `ControlGate` and `CompletionInterlock` so external write tools remain locked until plan approval.
- [ ] **Tool Cards & `/mcp` TUI controls:** Render external MCP tool calls as collapsible Tool Cards and inspect connected servers/tools via `/mcp`.

### 4.2 AllyH as an MCP Server (`allyh mcp serve`) (`allyh-tui`, `allyh-core`)
- [ ] **Add `allyh mcp serve` CLI subcommand (`crates/allyh-tui/src/cli.rs`):** Expose AllyH's core capabilities over stdio MCP for external editors/agents (VS Code, Cursor, Claude Code, Antigravity, Zed):
  - [ ] Memory tools: `query_invariants`, `record_decision`, `invalidate_decision`, `pin_decision`
  - [ ] RepoGraph tools: `inspect_symbol`, `get_ego_graph`, `get_env_delta`
  - [ ] Skills tools: `list_skills`, `get_skill`, `add_skill`
  - [ ] SIM / Task tools: `list_tasks`, `inspect_sim`
- [ ] **Respect `(repo_id, scope_id)` & bi-temporal conflict gates:** External clients writing architectural decisions go through the exact same LanceDB conflict gate and sleep-time consolidation pipeline.

### Milestone 4 Acceptance Criteria
- [ ] End-to-end MCP client/server round-trip tests proving an external MCP client can query bi-temporal invariants, retrieve polyglot RepoGraph ego-graphs, and trigger conflict-checked decision writes against AllyH's LanceDB store.

---

## Milestone 5: Provider Ecosystem, Prompt Caching & Cost-Aware SIM/CRSI (`v0.7.0`)

**Goal:** Broaden model provider choice, drastically reduce per-turn token cost using AllyH's deterministic preamble layout, and upgrade SIM/CRSI to optimize real-world coding metrics.

### 5.1 Expanded Providers & Prefix Prompt Caching (`allyh-core`)
- [ ] **Add native providers in `HybridRouter` (`crates/allyh-core/src/router.rs`):**
  - [ ] Google Gemini (`gemini:<model>`)
  - [ ] OpenRouter (`openrouter:<org/model>`)
  - [ ] OpenAI Responses API & reasoning effort controls (`--reasoning-effort low|medium|high`)
  - [ ] Optional feature-gated enterprise providers (AWS Bedrock / Vertex AI)
- [ ] **Cache-aware preamble partitioning (`crates/allyh-core/src/preamble.rs`):** Split `DynamicPreambleAssembler` output into a **stable prefix** (operating rules, pinned invariants, skills catalog, stable domain guidance) and a **volatile suffix** (`EnvDelta`, active task error trace, per-turn ego-graphs), emitting provider cache breakpoints (`cache_control` for Anthropic, prefix alignment for OpenAI/Gemini).
- [ ] **Token, cache-hit, and cost telemetry:** Track input/output/cached tokens and estimated session USD cost in `sessions` (LanceDB) and display them in the TUI status bar.

### 5.2 Polyglot & Cost-Aware SIM / CRSI (`allyh-sim`, `allyh-core`)
- [ ] **Multi-objective Selector in `allyh-sim` (`crates/allyh-sim/src/selector.rs`):** Incorporate token budget, tool-round efficiency, and latency alongside pass rate when ranking candidate `HarnessSpec` patches.
- [ ] **Polyglot benchmark suite presets (`crates/allyh-sim`):** Provide ready-to-run `EvalSuiteSpec` templates for Rust, TypeScript, Python, and Go tasks so users can run `allyh sim evolve` on their own repo's test suites.
- [ ] **Live session trace harvesting (`allyh sim harvest`):** Optionally distill real failed/recovered interactive TUI sessions (stored out-of-tree) into candidate SIM problem diagnoses (`P1..Pn`) during sleep-time compute.

### Milestone 5 Acceptance Criteria
- [ ] Unit tests verifying stable-prefix byte invariance across turns when only `EnvDelta` changes, accurate cache/cost accounting in `sessions`, and multi-objective SIM selector regression tests.

---

## Milestone 6: Headless CI Mode, Multi-Agent Worktrees & TUI Coherence Upgrades (`v0.8.0`)

**Goal:** Unlock automated pipelines and parallel multi-agent workflows built on AllyH's S3/object-store LanceDB backend, while elevating the interactive TUI.

### 6.1 Headless / CI Execution Mode (`allyh-tui`, `allyh-core`)
- [ ] **Add `allyh run` non-interactive subcommand:**
  - [ ] Execute a prompt or structured task spec headlessly with `--output jsonl` streaming tool records, diagnostics, and verification proofs.
  - [ ] Support `--mode discovery-only` (safe PR reviewer / architectural invariant auditor) and `--mode execute --auto-approve` (CI autofix with mandatory passing `VerificationProof`).
- [ ] **PR Architectural Drift Check (`allyh audit --base main`):** Compare a Git diff against active bi-temporal `architectural_decisions` in LanceDB and emit a structured violation report (exit code non-zero on invariant breach) for GitHub Actions / GitLab CI.

### 6.2 Multi-Agent Coordination over Shared LanceDB (`allyh-core`, `allyh-memory`, `allyh-env`)
- [ ] **Isolated `git worktree` execution per agent scope:** Spawn parallel scoped sub-agents (`scope_id = task-<id>`) in isolated Git worktrees backed by one shared LanceDB URI (`local` or `s3://`).
- [ ] **Bi-temporal Decision Promotion (`allyh memory promote --from <scope> --to team`):** Promote verified architectural decisions and newly discovered skills from an agent's private scope into a shared team scope through the conflict gate.
- [ ] **Cross-agent hash/claim awareness:** Warn in `EnvDelta` when another active scope sharing the same LanceDB URI has modified or claimed overlapping files.

### 6.3 TUI Coherence & Inspection Upgrades (`allyh-tui`)
- [ ] **Interactive Diff Modal:** Side-by-side / unified `syntect` diff viewer for proposed edits and checkpoint review.
- [ ] **Bi-Temporal Memory Timeline (`/timeline`):** Browse active *and* superseded (`invalid_at IS NOT NULL`) decisions, showing supersession chains (`superseded_by`) and rationale evolution.
- [ ] **Session Browser & Resume (`/sessions`):** List prior sessions for the repository, inspect rolling summaries, and resume or fork from a previous session cleanly.
- [ ] **Mouse scroll & copy-friendly Tool Cards:** Optional mouse-wheel scrolling in Chat/Inspector panes and one-key clipboard copy of code blocks or diagnostics.

### Milestone 6 Acceptance Criteria
- [ ] End-to-end headless CI test verifying `allyh audit` catches a deliberate architectural invariant violation on a diff, plus a multi-scope shared-LanceDB test verifying isolated worktree edits and conflict-gated decision promotion.

---

## Suggested Execution Order

1. **Start with Milestone 1 (`v0.3.0`)** — immediately unlocks AllyH for non-Rust codebases and polyglot monorepos while keeping `allyh-env` and `allyh-tools` cleanly decoupled.
2. **Follow with Milestone 2 (`v0.4.0`) + Milestone 3.1 (`search_code`)** — delivers the biggest jump in real-world coding accuracy (semantic RepoGraph + sub-second LSP diagnostics + fast code search).
3. **Ship Milestone 4 (`v0.6.0` MCP Server)** — multiplies adoption by letting developers use AllyH's bi-temporal memory and RepoGraph inside existing editors immediately.
4. **Iterate through Milestones 5 & 6** — optimize cost/speed with prompt caching + cost-aware CRSI, then expand into CI drift auditing (`allyh audit`) and multi-agent swarms.

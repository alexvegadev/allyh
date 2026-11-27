use std::collections::{BTreeMap, HashSet};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{MemoryError, Result};
use crate::schema::{SimAttemptRecord, SimMemoryMode, SimProblemRecord, SimRunRecord};

/// The 8 SIM conditioning rules for Proposer and Selector prompts (arXiv:2611.18856 Appendix D.2).
pub const SIM_PROMPT_RULES: &[&str] = &[
    "1. Treat SIM entries as prior experience, with flexibility to choose a different edit.",
    "2. Check the domain, benchmark, and policy model provenance behind each entry.",
    "3. Start with an existing diagnosis when it explains the current failure.",
    "4. Reuse the rationale and reassess the edit after a benchmark or model change.",
    "5. Revise, replace, or override a harness modification when new evidence warrants it.",
    "6. Revisit a tested direction only when new evidence justifies it.",
    "7. Create a new problem entry when existing diagnoses do not explain the failure.",
    "8. Give task-specific evidence more weight in similar settings and less when both benchmark and model change.",
];

/// Carryover lineage metadata matching Appendix D.3 `carried_from`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SimCarriedFrom {
    pub arm: String,
    pub domain: String,
    pub into: String,
}

/// Portable Appendix D.3 attempt record inside a [`SimProblemEntry`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SimAttemptEntry {
    pub round: u32,
    pub sample: String,
    pub accepted: bool,
    #[serde(default)]
    pub backfilled: bool,
    pub what: String,
    #[serde(default)]
    pub files: Vec<String>,
    #[serde(default)]
    pub predicted: Vec<String>,
    #[serde(default)]
    pub fixed: Vec<String>,
    #[serde(default)]
    pub missed: Vec<String>,
    pub passed: u32,
    pub total: u32,
    #[serde(default)]
    pub score: f32,
    pub domain: String,
    #[serde(default)]
    pub benchmark: String,
    #[serde(default)]
    pub policy_model: String,
    #[serde(default)]
    pub harness_version_id: String,
    #[serde(default = "Utc::now")]
    pub created_at: DateTime<Utc>,
}

/// Problem entry m_i = (c_i, [(a_ij, o_ij, p_ij)]_j) matching Appendix D.3.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SimProblemEntry {
    pub id: String,
    pub name: String,
    pub opened_round: u32,
    #[serde(default)]
    pub origin_domain: String,
    #[serde(default)]
    pub origin_run_id: String,
    #[serde(default)]
    pub attempts: Vec<SimAttemptEntry>,
}

/// Canonical Appendix D.3 portable JSON document for SIM export/import.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SimExportDocument {
    pub next_id: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub carried_from: Option<SimCarriedFrom>,
    #[serde(default)]
    pub aliases: BTreeMap<String, String>,
    #[serde(default)]
    pub problems: Vec<SimProblemEntry>,
}

/// Proposer's explicit problem match choice (Appendix D.1: proposer decides, no separate similarity model).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum SimProblemRef {
    /// Cite an existing problem identifier (e.g. `P3` or a merged alias like `P6`).
    Existing(String),
    /// Open a new root-cause hypothesis (deterministically merged if exact text already exists).
    New(String),
}

/// Input submitted to `U(M_t, e_t)` when the selector accepts a candidate modification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimAttemptInput {
    pub problem_ref: SimProblemRef,
    pub round: u32,
    pub sample: String,
    pub accepted: bool,
    pub backfilled: bool,
    pub what: String,
    pub files: Vec<String>,
    pub predicted: Vec<String>,
    pub fixed: Vec<String>,
    pub missed: Vec<String>,
    pub passed: u32,
    pub total: u32,
    pub score: f32,
    pub domain: String,
    pub benchmark: String,
    pub policy_model: String,
    pub harness_version_id: String,
}

/// Outcome returned after applying an accepted edit `e_t` to a [`SimLedger`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimUpdateResult {
    pub canonical_problem_id: String,
    pub opened_new_problem: bool,
    pub merged_from_alias: Option<String>,
    pub attempt_sequence_idx: u32,
}

/// Full in-memory Self-Improvement Memory ledger for a CRSI run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimLedger {
    pub run: SimRunRecord,
    pub aliases: BTreeMap<String, String>,
    pub problems: Vec<SimProblemEntry>,
}

impl SimLedger {
    /// Start a fresh source RSI run with empty SIM `(H_0, ∅)`.
    #[allow(clippy::too_many_arguments)]
    pub fn new_source(
        repo_id: impl Into<String>,
        scope_id: impl Into<String>,
        arm_name: impl Into<String>,
        domain: impl Into<String>,
        benchmark: impl Into<String>,
        policy_model: impl Into<String>,
        optimizer_model: impl Into<String>,
        seed_harness_id: impl Into<String>,
        memory_mode: SimMemoryMode,
    ) -> Self {
        let now = Utc::now();
        let seed_harness_id = seed_harness_id.into();
        let run = SimRunRecord {
            run_id: Uuid::new_v4().to_string(),
            repo_id: repo_id.into(),
            scope_id: scope_id.into(),
            arm_name: arm_name.into(),
            domain: domain.into(),
            benchmark: benchmark.into(),
            policy_model: policy_model.into(),
            optimizer_model: optimizer_model.into(),
            memory_mode,
            seed_harness_id: seed_harness_id.clone(),
            current_harness_id: seed_harness_id,
            carried_from_run_id: None,
            carried_from_domain: None,
            next_problem_num: 1,
            aliases_json: "{}".to_string(),
            rounds_completed: 0,
            created_at: now,
            updated_at: now,
        };
        Self {
            run,
            aliases: BTreeMap::new(),
            problems: Vec::new(),
        }
    }

    /// Create a target CRSI continuation ledger from this source ledger under the chosen [`SimMemoryMode`].
    ///
    /// - `ColdStart`: empty ledger starting from `target_seed_harness_id` (no carryover).
    /// - `HarnessTransfer`: empty ledger starting from `carried_harness_id` (`(H_S, ∅)`).
    /// - `ProblemOnly`: carries root-cause hypotheses `c_i` only (`(H_S, M_S^c)`), stripping attempt histories.
    /// - `FullSim`: carries all problems, ordered attempts, aliases, and `carried_from` provenance (`(H_S, M_S)`).
    #[allow(clippy::too_many_arguments)]
    pub fn carry_into(
        &self,
        target_repo_id: impl Into<String>,
        target_scope_id: impl Into<String>,
        target_arm_name: impl Into<String>,
        target_domain: impl Into<String>,
        target_benchmark: impl Into<String>,
        target_policy_model: impl Into<String>,
        target_optimizer_model: impl Into<String>,
        starting_harness_id: impl Into<String>,
        mode: SimMemoryMode,
    ) -> Self {
        let now = Utc::now();
        let starting_harness_id = starting_harness_id.into();
        let (problems, aliases, next_problem_num, carried_from_run_id, carried_from_domain) =
            match mode {
                SimMemoryMode::ColdStart | SimMemoryMode::HarnessTransfer => (
                    Vec::new(),
                    BTreeMap::new(),
                    1,
                    if mode == SimMemoryMode::HarnessTransfer {
                        Some(self.run.run_id.clone())
                    } else {
                        None
                    },
                    if mode == SimMemoryMode::HarnessTransfer {
                        Some(self.run.domain.clone())
                    } else {
                        None
                    },
                ),
                SimMemoryMode::ProblemOnly => {
                    let stripped = self
                        .problems
                        .iter()
                        .map(|p| SimProblemEntry {
                            id: p.id.clone(),
                            name: p.name.clone(),
                            opened_round: p.opened_round,
                            origin_domain: p.origin_domain.clone(),
                            origin_run_id: p.origin_run_id.clone(),
                            attempts: Vec::new(),
                        })
                        .collect();
                    (
                        stripped,
                        self.aliases.clone(),
                        self.run.next_problem_num.max(1),
                        Some(self.run.run_id.clone()),
                        Some(self.run.domain.clone()),
                    )
                }
                SimMemoryMode::FullSim => (
                    self.problems.clone(),
                    self.aliases.clone(),
                    self.run.next_problem_num.max(1),
                    Some(self.run.run_id.clone()),
                    Some(self.run.domain.clone()),
                ),
            };

        let aliases_json = serde_json::to_string(&aliases).unwrap_or_else(|_| "{}".to_string());
        let run = SimRunRecord {
            run_id: Uuid::new_v4().to_string(),
            repo_id: target_repo_id.into(),
            scope_id: target_scope_id.into(),
            arm_name: target_arm_name.into(),
            domain: target_domain.into(),
            benchmark: target_benchmark.into(),
            policy_model: target_policy_model.into(),
            optimizer_model: target_optimizer_model.into(),
            memory_mode: mode,
            seed_harness_id: starting_harness_id.clone(),
            current_harness_id: starting_harness_id,
            carried_from_run_id,
            carried_from_domain,
            next_problem_num,
            aliases_json,
            rounds_completed: 0,
            created_at: now,
            updated_at: now,
        };
        Self {
            run,
            aliases,
            problems,
        }
    }

    /// Resolve a problem identifier transitively through `aliases` (cycle-safe).
    pub fn resolve_alias(&self, problem_id: &str) -> String {
        let mut current = problem_id.trim().to_ascii_uppercase();
        let mut visited = HashSet::new();
        while let Some(target) = self.aliases.get(&current) {
            if !visited.insert(current.clone()) {
                break;
            }
            current = target.clone();
        }
        current
    }

    /// Look up a problem entry by ID or alias.
    pub fn get_problem(&self, problem_id: &str) -> Option<&SimProblemEntry> {
        let canonical = self.resolve_alias(problem_id);
        self.problems
            .iter()
            .find(|p| p.id.eq_ignore_ascii_case(&canonical))
    }

    /// Deterministic SIM update `U(M_t, e_t)` (§3.2 & Appendix D.1).
    ///
    /// Only accepted (or explicitly backfilled) attempts are recorded. When `memory_mode` is
    /// `ColdStart` or `HarnessTransfer`, callers generally do not record SIM entries; if invoked,
    /// this method still honors explicit recording so ablation runners can inspect/record if desired.
    pub fn record_accepted_attempt(&mut self, input: SimAttemptInput) -> Result<SimUpdateResult> {
        if !input.accepted && !input.backfilled {
            return Err(MemoryError::InvalidSim(
                "SIM only persists selector-accepted (or backfilled) attempts; rejected candidates leave M_t unchanged".to_string(),
            ));
        }

        let now = Utc::now();
        let attempt = SimAttemptEntry {
            round: input.round,
            sample: input.sample,
            accepted: input.accepted,
            backfilled: input.backfilled,
            what: input.what.trim().to_string(),
            files: input.files,
            predicted: input.predicted,
            fixed: input.fixed,
            missed: input.missed,
            passed: input.passed,
            total: input.total,
            score: input.score,
            domain: input.domain.clone(),
            benchmark: input.benchmark,
            policy_model: input.policy_model,
            harness_version_id: input.harness_version_id.clone(),
            created_at: now,
        };

        let mut merged_from_alias = None;
        let (canonical_problem_id, opened_new_problem, attempt_sequence_idx) = match input.problem_ref
        {
            SimProblemRef::Existing(raw_id) => {
                let raw_upper = raw_id.trim().to_ascii_uppercase();
                let canonical = self.resolve_alias(&raw_upper);
                if canonical != raw_upper {
                    merged_from_alias = Some(raw_upper);
                }
                let idx = self
                    .problems
                    .iter()
                    .position(|p| p.id.eq_ignore_ascii_case(&canonical))
                    .ok_or_else(|| {
                        MemoryError::InvalidSim(format!(
                            "cited SIM problem '{raw_id}' (resolved: '{canonical}') does not exist"
                        ))
                    })?;
                let seq = self.problems[idx].attempts.len() as u32;
                if self.run.memory_mode != SimMemoryMode::ProblemOnly {
                    self.problems[idx].attempts.push(attempt);
                }
                (self.problems[idx].id.clone(), false, seq)
            }
            SimProblemRef::New(raw_name) => {
                let name = raw_name.trim();
                if name.is_empty() {
                    return Err(MemoryError::InvalidSim(
                        "new SIM problem root-cause hypothesis must not be empty".to_string(),
                    ));
                }
                let normalized = normalize_root_cause(name);
                // Deterministic exact-text matching: reuse existing entry if root-cause text matches.
                if let Some(existing_idx) = self
                    .problems
                    .iter()
                    .position(|p| normalize_root_cause(&p.name) == normalized)
                {
                    let seq = self.problems[existing_idx].attempts.len() as u32;
                    if self.run.memory_mode != SimMemoryMode::ProblemOnly {
                        self.problems[existing_idx].attempts.push(attempt);
                    }
                    (self.problems[existing_idx].id.clone(), false, seq)
                } else {
                    let id = format!("P{}", self.run.next_problem_num.max(1));
                    self.run.next_problem_num = self.run.next_problem_num.max(1) + 1;
                    let attempts = if self.run.memory_mode == SimMemoryMode::ProblemOnly {
                        Vec::new()
                    } else {
                        vec![attempt]
                    };
                    self.problems.push(SimProblemEntry {
                        id: id.clone(),
                        name: name.to_string(),
                        opened_round: input.round,
                        origin_domain: input.domain,
                        origin_run_id: self.run.run_id.clone(),
                        attempts,
                    });
                    (id, true, 0)
                }
            }
        };

        self.run.current_harness_id = input.harness_version_id;
        self.run.updated_at = now;
        self.sync_aliases_json();

        Ok(SimUpdateResult {
            canonical_problem_id,
            opened_new_problem,
            merged_from_alias,
            attempt_sequence_idx,
        })
    }

    /// Deterministically merge any problem entries with identical normalized root-cause text
    /// without dropping any historical attempts, recording `aliases[dup_id] = canonical_id`.
    /// Returns the list of `(merged_alias_id, canonical_id)` pairs created.
    pub fn merge_identical_problems(&mut self) -> Vec<(String, String)> {
        let mut seen: BTreeMap<String, usize> = BTreeMap::new();
        let mut merged_pairs = Vec::new();
        let mut kept: Vec<SimProblemEntry> = Vec::with_capacity(self.problems.len());

        for entry in std::mem::take(&mut self.problems) {
            let norm = normalize_root_cause(&entry.name);
            if let Some(&canonical_idx) = seen.get(&norm) {
                let canonical_id = kept[canonical_idx].id.clone();
                let dup_id = entry.id.clone();
                if dup_id != canonical_id {
                    self.aliases.insert(dup_id.clone(), canonical_id.clone());
                    merged_pairs.push((dup_id, canonical_id));
                }
                kept[canonical_idx].attempts.extend(entry.attempts);
                kept[canonical_idx].opened_round =
                    kept[canonical_idx].opened_round.min(entry.opened_round);
            } else {
                let idx = kept.len();
                seen.insert(norm, idx);
                kept.push(entry);
            }
        }

        // Flatten transitive alias chains so every alias points directly to its canonical target.
        let keys: Vec<String> = self.aliases.keys().cloned().collect();
        for key in keys {
            let resolved = self.resolve_alias(&key);
            self.aliases.insert(key, resolved);
        }

        self.problems = kept;
        self.sync_aliases_json();
        merged_pairs
    }

    /// Export this ledger to the exact Appendix D.3 canonical JSON schema.
    pub fn to_export_document(&self) -> SimExportDocument {
        let carried_from = match (
            &self.run.carried_from_run_id,
            &self.run.carried_from_domain,
        ) {
            (Some(arm), Some(domain)) => Some(SimCarriedFrom {
                arm: arm.clone(),
                domain: domain.clone(),
                into: self.run.arm_name.clone(),
            }),
            _ => None,
        };
        SimExportDocument {
            next_id: self.run.next_problem_num.max(1),
            carried_from,
            aliases: self.aliases.clone(),
            problems: self.problems.clone(),
        }
    }

    /// Reconstruct a [`SimLedger`] from a portable Appendix D.3 JSON document.
    #[allow(clippy::too_many_arguments)]
    pub fn from_export_document(
        doc: SimExportDocument,
        repo_id: impl Into<String>,
        scope_id: impl Into<String>,
        arm_name: impl Into<String>,
        domain: impl Into<String>,
        benchmark: impl Into<String>,
        policy_model: impl Into<String>,
        optimizer_model: impl Into<String>,
        harness_version_id: impl Into<String>,
    ) -> Result<Self> {
        let now = Utc::now();
        let run_id = Uuid::new_v4().to_string();
        let domain_str = domain.into();
        let max_existing = doc
            .problems
            .iter()
            .filter_map(|p| p.id.strip_prefix('P').and_then(|n| n.parse::<u32>().ok()))
            .max()
            .unwrap_or(0);
        let next_problem_num = doc.next_id.max(max_existing + 1).max(1);
        let aliases_json = serde_json::to_string(&doc.aliases)?;
        let harness_version_id = harness_version_id.into();

        let (carried_from_run_id, carried_from_domain) = match doc.carried_from {
            Some(cf) => (Some(cf.arm), Some(cf.domain)),
            None => (None, None),
        };

        let problems = doc
            .problems
            .into_iter()
            .map(|mut p| {
                if p.origin_domain.is_empty() {
                    p.origin_domain = p
                        .attempts
                        .first()
                        .map(|a| a.domain.clone())
                        .filter(|d| !d.is_empty())
                        .unwrap_or_else(|| domain_str.clone());
                }
                if p.origin_run_id.is_empty() {
                    p.origin_run_id = carried_from_run_id
                        .clone()
                        .unwrap_or_else(|| run_id.clone());
                }
                p
            })
            .collect();

        let run = SimRunRecord {
            run_id,
            repo_id: repo_id.into(),
            scope_id: scope_id.into(),
            arm_name: arm_name.into(),
            domain: domain_str,
            benchmark: benchmark.into(),
            policy_model: policy_model.into(),
            optimizer_model: optimizer_model.into(),
            memory_mode: SimMemoryMode::FullSim,
            seed_harness_id: harness_version_id.clone(),
            current_harness_id: harness_version_id,
            carried_from_run_id,
            carried_from_domain,
            next_problem_num,
            aliases_json,
            rounds_completed: 0,
            created_at: now,
            updated_at: now,
        };

        let mut ledger = Self {
            run,
            aliases: doc.aliases,
            problems,
        };
        ledger.merge_identical_problems();
        Ok(ledger)
    }

    /// Render the **complete** SIM ledger (zero truncation, zero lossy retrieval) for Proposer
    /// and Selector system prompts under the specified [`SimMemoryMode`].
    pub fn render_for_prompt(&self, mode: SimMemoryMode) -> String {
        match mode {
            SimMemoryMode::ColdStart | SimMemoryMode::HarnessTransfer => {
                "## Self-Improvement Memory (SIM)\n(Disabled for this memory condition.)"
                    .to_string()
            }
            SimMemoryMode::ProblemOnly => {
                if self.problems.is_empty() {
                    return "## Self-Improvement Memory (SIM — Root Causes Only)\n(No recorded harness problems yet.)".to_string();
                }
                let mut lines = vec![
                    "## Self-Improvement Memory (SIM — Root Causes Only)".to_string(),
                ];
                for p in &self.problems {
                    lines.push(format!("- [{}] {}", p.id, p.name));
                }
                lines.join("\n")
            }
            SimMemoryMode::FullSim => {
                let mut lines = vec!["## Self-Improvement Memory (SIM — Complete Ledger)".to_string()];
                if let (Some(from_arm), Some(from_dom)) =
                    (&self.run.carried_from_run_id, &self.run.carried_from_domain)
                {
                    lines.push(format!(
                        "Carried from: `{from_arm}` (domain: `{from_dom}`) -> `{}` (domain: `{}`, policy: `{}`)",
                        self.run.arm_name, self.run.domain, self.run.policy_model
                    ));
                }
                if !self.aliases.is_empty() {
                    let alias_list: Vec<String> = self
                        .aliases
                        .iter()
                        .map(|(k, v)| format!("{k}->{v}"))
                        .collect();
                    lines.push(format!("Merged aliases: {}", alias_list.join(", ")));
                }
                if self.problems.is_empty() {
                    lines.push("(No recorded harness problems yet.)".to_string());
                    return lines.join("\n");
                }
                for p in &self.problems {
                    lines.push(format!(
                        "\n### {} (opened round {}, origin: `{}`): {}",
                        p.id, p.opened_round, p.origin_domain, p.name
                    ));
                    if p.attempts.is_empty() {
                        lines.push("  (No recorded attempts yet.)".to_string());
                    }
                    for a in &p.attempts {
                        let status = if a.accepted { "accepted" } else { "backfilled" };
                        lines.push(format!(
                            "  - Round {} [{}] ({status}, domain=`{}`, benchmark=`{}`, policy=`{}`, passed={}/{}, score={:.2})",
                            a.round,
                            a.sample,
                            a.domain,
                            a.benchmark,
                            a.policy_model,
                            a.passed,
                            a.total,
                            a.score
                        ));
                        lines.push(format!("    What: {}", a.what));
                        if !a.files.is_empty() {
                            lines.push(format!("    Files: {}", a.files.join(", ")));
                        }
                        lines.push(format!(
                            "    Predicted: [{}] | Fixed: [{}] | Missed: [{}]",
                            a.predicted.join(", "),
                            a.fixed.join(", "),
                            a.missed.join(", ")
                        ));
                    }
                }
                lines.join("\n")
            }
        }
    }

    /// Flatten this ledger into normalized LanceDB rows (`SimRunRecord`, `SimProblemRecord`s, `SimAttemptRecord`s).
    pub fn to_records(&self) -> (SimRunRecord, Vec<SimProblemRecord>, Vec<SimAttemptRecord>) {
        let mut run = self.run.clone();
        run.aliases_json = serde_json::to_string(&self.aliases).unwrap_or_else(|_| "{}".into());

        let mut problems = Vec::with_capacity(self.problems.len());
        let mut attempts = Vec::new();

        for p in &self.problems {
            let row_id = format!("{}:{}", run.run_id, p.id);
            let origin_domain = if p.origin_domain.is_empty() {
                run.domain.clone()
            } else {
                p.origin_domain.clone()
            };
            let origin_run_id = if p.origin_run_id.is_empty() {
                run.run_id.clone()
            } else {
                p.origin_run_id.clone()
            };
            problems.push(SimProblemRecord {
                row_id,
                run_id: run.run_id.clone(),
                repo_id: run.repo_id.clone(),
                scope_id: run.scope_id.clone(),
                problem_id: p.id.clone(),
                name: p.name.clone(),
                normalized_name: normalize_root_cause(&p.name),
                opened_round: p.opened_round,
                origin_domain,
                origin_run_id,
                created_at: p
                    .attempts
                    .first()
                    .map(|a| a.created_at)
                    .unwrap_or(run.created_at),
            });

            for (seq, a) in p.attempts.iter().enumerate() {
                attempts.push(SimAttemptRecord {
                    id: format!("{}:{}:{seq}", run.run_id, p.id),
                    run_id: run.run_id.clone(),
                    repo_id: run.repo_id.clone(),
                    scope_id: run.scope_id.clone(),
                    problem_id: p.id.clone(),
                    sequence_idx: seq as u32,
                    round: a.round,
                    sample: a.sample.clone(),
                    accepted: a.accepted,
                    backfilled: a.backfilled,
                    what: a.what.clone(),
                    files: a.files.clone(),
                    predicted: a.predicted.clone(),
                    fixed: a.fixed.clone(),
                    missed: a.missed.clone(),
                    passed: a.passed,
                    total: a.total,
                    score: a.score,
                    domain: a.domain.clone(),
                    benchmark: a.benchmark.clone(),
                    policy_model: a.policy_model.clone(),
                    harness_version_id: a.harness_version_id.clone(),
                    created_at: a.created_at,
                });
            }
        }

        (run, problems, attempts)
    }

    /// Reconstruct a [`SimLedger`] from LanceDB rows.
    pub fn from_records(
        run: SimRunRecord,
        mut problem_rows: Vec<SimProblemRecord>,
        mut attempt_rows: Vec<SimAttemptRecord>,
    ) -> Result<Self> {
        let aliases: BTreeMap<String, String> = if run.aliases_json.trim().is_empty() {
            BTreeMap::new()
        } else {
            serde_json::from_str(&run.aliases_json)?
        };

        problem_rows.sort_by_key(|p| {
            p.problem_id
                .strip_prefix('P')
                .and_then(|n| n.parse::<u32>().ok())
                .unwrap_or(u32::MAX)
        });
        attempt_rows.sort_by_key(|a| (a.sequence_idx, a.round));

        let mut attempts_by_problem: BTreeMap<String, Vec<SimAttemptEntry>> = BTreeMap::new();
        for a in attempt_rows {
            attempts_by_problem
                .entry(a.problem_id)
                .or_default()
                .push(SimAttemptEntry {
                    round: a.round,
                    sample: a.sample,
                    accepted: a.accepted,
                    backfilled: a.backfilled,
                    what: a.what,
                    files: a.files,
                    predicted: a.predicted,
                    fixed: a.fixed,
                    missed: a.missed,
                    passed: a.passed,
                    total: a.total,
                    score: a.score,
                    domain: a.domain,
                    benchmark: a.benchmark,
                    policy_model: a.policy_model,
                    harness_version_id: a.harness_version_id,
                    created_at: a.created_at,
                });
        }

        let problems = problem_rows
            .into_iter()
            .map(|p| {
                let attempts = attempts_by_problem.remove(&p.problem_id).unwrap_or_default();
                SimProblemEntry {
                    id: p.problem_id,
                    name: p.name,
                    opened_round: p.opened_round,
                    origin_domain: p.origin_domain,
                    origin_run_id: p.origin_run_id,
                    attempts,
                }
            })
            .collect();

        Ok(Self {
            run,
            aliases,
            problems,
        })
    }

    /// Count total attempts across all problems in this ledger.
    pub fn total_attempts(&self) -> usize {
        self.problems.iter().map(|p| p.attempts.len()).sum()
    }

    fn sync_aliases_json(&mut self) {
        self.run.aliases_json =
            serde_json::to_string(&self.aliases).unwrap_or_else(|_| "{}".to_string());
    }
}

/// Normalize a root-cause hypothesis for deterministic exact-text matching (Appendix D.1).
pub fn normalize_root_cause(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
}

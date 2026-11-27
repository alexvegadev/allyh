use allyh_memory::SimProblemRef;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Single candidate harness edit proposed in round `t` before evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateProposal {
    /// Sample identifier within the round (e.g. `s1`, `s2`, ...).
    pub sample_id: String,
    /// Proposer's explicit match choice: existing problem ID or new root-cause hypothesis (Appendix D.1).
    pub problem_ref: SimProblemRef,
    /// Modification description and rationale (`a_ij`).
    pub what: String,
    /// Harness components or files touched by this edit.
    #[serde(default)]
    pub files: Vec<String>,
    /// Ex-ante list of evolve task IDs expected to be fixed by this edit (locked before evaluation!).
    #[serde(default)]
    pub predicted: Vec<String>,
    /// RFC 7396 JSON Merge Patch applied onto the current `HarnessSpec`.
    pub patch: serde_json::Value,
}

/// Pre-evaluation change manifest locking each candidate's cited problem and predictions (Appendix D.1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeManifest {
    pub round: u32,
    pub base_harness_id: String,
    pub candidates: Vec<CandidateProposal>,
    pub created_at: DateTime<Utc>,
}

/// Single round record inside the ordinary target-run history `D_t` (distinct from SIM `M_t`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundHistoryEntry {
    pub round: u32,
    pub baseline_score: f32,
    pub accepted_sample: Option<String>,
    pub accepted_what: Option<String>,
    pub resulting_score: f32,
    pub notes: String,
}

/// Ordinary within-run history `D_t` retained across all memory conditions (§3.3 & Appendix C.3).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunHistory {
    pub entries: Vec<RoundHistoryEntry>,
}

impl RunHistory {
    pub fn push(&mut self, entry: RoundHistoryEntry) {
        self.entries.push(entry);
    }

    pub fn render_for_prompt(&self) -> String {
        if self.entries.is_empty() {
            return "## Current Run History (D_t)\n(First round of this run.)".to_string();
        }
        let mut lines = vec!["## Current Run History (D_t)".to_string()];
        for e in &self.entries {
            let outcome = match (&e.accepted_sample, &e.accepted_what) {
                (Some(s), Some(what)) => format!("accepted {s}: {what}"),
                _ => "no edit accepted".to_string(),
            };
            lines.push(format!(
                "- Round {}: {:.2} -> {:.2} ({outcome}) — {}",
                e.round, e.baseline_score, e.resulting_score, e.notes
            ));
        }
        lines.join("\n")
    }
}

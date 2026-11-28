use std::collections::BTreeMap;

use crate::error::{Result, SimError};
use crate::eval::{EvalSuiteSpec, EvalTask};

/// Supported built-in polyglot SIM benchmark suite preset names.
pub const SUITE_PRESET_NAMES: &[&str] = &["rust", "typescript", "python", "go", "polyglot"];

/// List available polyglot benchmark preset identifiers.
pub fn list_suite_presets() -> &'static [&'static str] {
    SUITE_PRESET_NAMES
}

/// Build a ready-to-run [`EvalSuiteSpec`] template for `rust`, `typescript`, `python`, `go`,
/// or `polyglot` (all languages combined) so users can immediately run `allyh sim evolve`.
pub fn polyglot_suite_preset(name: &str) -> Result<EvalSuiteSpec> {
    match name.trim().to_ascii_lowercase().as_str() {
        "rust" | "rs" => Ok(rust_preset()),
        "typescript" | "ts" => Ok(typescript_preset()),
        "python" | "py" => Ok(python_preset()),
        "go" | "golang" => Ok(go_preset()),
        "polyglot" | "all" => Ok(polyglot_combined_preset()),
        other => Err(SimError::InvalidSuite(format!(
            "unknown suite preset '{other}' (available: {})",
            SUITE_PRESET_NAMES.join(", ")
        ))),
    }
}

fn rust_preset() -> EvalSuiteSpec {
    EvalSuiteSpec {
        name: "allyh-rust-preset".into(),
        domain: "rust".into(),
        benchmark: "rust-coherence-v1".into(),
        split_seed: 20_261_113,
        tasks: vec![
            EvalTask {
                id: "rs-error-enum".into(),
                title: "Refactor library error enum with thiserror".into(),
                prompt: "Inspect src/lib.rs and explain how to expose structured ParseError variants cleanly.".into(),
                setup_files: BTreeMap::from([(
                    "src/lib.rs".into(),
                    "pub fn parse_port(raw: &str) -> Result<u16, String> {\n    raw.parse::<u16>().map_err(|e| e.to_string())\n}\n".into(),
                )]),
                verifier_command: None,
                expected_substrings: vec!["parse_port".into()],
                domain_tags: vec!["rust".into(), "needs_contract_guidelines".into()],
            },
            EvalTask {
                id: "rs-borrow-check".into(),
                title: "Resolve mutable borrow conflict cleanly".into(),
                prompt: "Inspect src/cache.rs and identify how to avoid simultaneous mutable/immutable borrows.".into(),
                setup_files: BTreeMap::from([(
                    "src/cache.rs".into(),
                    "use std::collections::HashMap;\npub fn get_or_insert(map: &mut HashMap<String, usize>, k: &str) -> usize {\n    *map.entry(k.to_string()).or_insert(0)\n}\n".into(),
                )]),
                verifier_command: None,
                expected_substrings: vec!["get_or_insert".into()],
                domain_tags: vec!["rust".into(), "needs_interlock".into()],
            },
            EvalTask {
                id: "rs-async-channel".into(),
                title: "Audit bounded channel backpressure".into(),
                prompt: "Review src/worker.rs and verify bounded queue handling under shutdown.".into(),
                setup_files: BTreeMap::from([(
                    "src/worker.rs".into(),
                    "pub struct WorkerConfig { pub capacity: usize }\npub fn default_capacity() -> usize { 64 }\n".into(),
                )]),
                verifier_command: None,
                expected_substrings: vec!["WorkerConfig".into()],
                domain_tags: vec!["rust".into(), "needs_verification_proof".into()],
            },
            EvalTask {
                id: "rs-hash-edit".into(),
                title: "Verify stale-hash edit safety".into(),
                prompt: "Inspect src/config.rs before making changes so SHA-256 hashes stay synchronized.".into(),
                setup_files: BTreeMap::from([(
                    "src/config.rs".into(),
                    "pub const DEFAULT_TIMEOUT_SECS: u64 = 30;\n".into(),
                )]),
                verifier_command: None,
                expected_substrings: vec!["DEFAULT_TIMEOUT_SECS".into()],
                domain_tags: vec!["rust".into(), "needs_stale_hash_hint".into()],
            },
        ],
    }
}

fn typescript_preset() -> EvalSuiteSpec {
    EvalSuiteSpec {
        name: "allyh-typescript-preset".into(),
        domain: "typescript".into(),
        benchmark: "ts-coherence-v1".into(),
        split_seed: 20_261_114,
        tasks: vec![
            EvalTask {
                id: "ts-strict-null".into(),
                title: "Guard nullable lookup in pricing module".into(),
                prompt: "Inspect src/pricing.ts and explain how computeDiscount handles missing coupons safely.".into(),
                setup_files: BTreeMap::from([(
                    "src/pricing.ts".into(),
                    "export interface Coupon { code: string; pct: number }\nexport function computeDiscount(price: number, coupon?: Coupon): number {\n  return coupon ? price * (1 - coupon.pct) : price;\n}\n".into(),
                )]),
                verifier_command: None,
                expected_substrings: vec!["computeDiscount".into()],
                domain_tags: vec!["typescript".into(), "needs_contract_guidelines".into()],
            },
            EvalTask {
                id: "ts-discriminated-union".into(),
                title: "Exhaustive switch over API response union".into(),
                prompt: "Inspect src/api.ts and confirm exhaustive status handling for ApiResult.".into(),
                setup_files: BTreeMap::from([(
                    "src/api.ts".into(),
                    "export type ApiResult<T> = { ok: true; data: T } | { ok: false; error: string };\nexport function unwrapOr<T>(res: ApiResult<T>, fallback: T): T {\n  return res.ok ? res.data : fallback;\n}\n".into(),
                )]),
                verifier_command: None,
                expected_substrings: vec!["unwrapOr".into()],
                domain_tags: vec!["typescript".into(), "needs_interlock".into()],
            },
            EvalTask {
                id: "ts-async-retry".into(),
                title: "Check retry backoff bounds".into(),
                prompt: "Inspect src/retry.ts and verify maxAttempts guards against infinite loops.".into(),
                setup_files: BTreeMap::from([(
                    "src/retry.ts".into(),
                    "export function backoffMs(attempt: number): number {\n  return Math.min(30_000, 200 * 2 ** Math.max(0, attempt));\n}\n".into(),
                )]),
                verifier_command: None,
                expected_substrings: vec!["backoffMs".into()],
                domain_tags: vec!["typescript".into(), "needs_verification_proof".into()],
            },
            EvalTask {
                id: "ts-schema-parse".into(),
                title: "Validate external config payload".into(),
                prompt: "Inspect src/env.ts and verify required fields are validated before startup.".into(),
                setup_files: BTreeMap::from([(
                    "src/env.ts".into(),
                    "export function requireEnv(map: Record<string, string | undefined>, key: string): string {\n  const v = map[key];\n  if (!v) throw new Error(`Missing ${key}`);\n  return v;\n}\n".into(),
                )]),
                verifier_command: None,
                expected_substrings: vec!["requireEnv".into()],
                domain_tags: vec!["typescript".into(), "needs_stale_hash_hint".into()],
            },
        ],
    }
}

fn python_preset() -> EvalSuiteSpec {
    EvalSuiteSpec {
        name: "allyh-python-preset".into(),
        domain: "python".into(),
        benchmark: "py-coherence-v1".into(),
        split_seed: 20_261_115,
        tasks: vec![
            EvalTask {
                id: "py-type-annotations".into(),
                title: "Preserve strict type annotations in ledger".into(),
                prompt: "Inspect app/ledger.py and check summarize_totals type safety.".into(),
                setup_files: BTreeMap::from([(
                    "app/ledger.py".into(),
                    "from typing import Iterable\n\ndef summarize_totals(values: Iterable[float]) -> float:\n    return round(sum(values), 2)\n".into(),
                )]),
                verifier_command: None,
                expected_substrings: vec!["summarize_totals".into()],
                domain_tags: vec!["python".into(), "needs_contract_guidelines".into()],
            },
            EvalTask {
                id: "py-mutable-default".into(),
                title: "Avoid mutable default argument bug".into(),
                prompt: "Inspect app/tags.py and verify append_tag does not share default list state.".into(),
                setup_files: BTreeMap::from([(
                    "app/tags.py".into(),
                    "def append_tag(tag: str, existing: list[str] | None = None) -> list[str]:\n    out = list(existing) if existing else []\n    out.append(tag)\n    return out\n".into(),
                )]),
                verifier_command: None,
                expected_substrings: vec!["append_tag".into()],
                domain_tags: vec!["python".into(), "needs_interlock".into()],
            },
            EvalTask {
                id: "py-path-safety".into(),
                title: "Prevent path traversal in loader".into(),
                prompt: "Inspect app/loader.py and confirm resolve_safe_path stays inside root.".into(),
                setup_files: BTreeMap::from([(
                    "app/loader.py".into(),
                    "from pathlib import Path\n\ndef resolve_safe_path(root: Path, rel: str) -> Path:\n    target = (root / rel).resolve()\n    target.relative_to(root.resolve())\n    return target\n".into(),
                )]),
                verifier_command: None,
                expected_substrings: vec!["resolve_safe_path".into()],
                domain_tags: vec!["python".into(), "needs_verification_proof".into()],
            },
            EvalTask {
                id: "py-pagination".into(),
                title: "Guard zero-limit pagination edge case".into(),
                prompt: "Inspect app/paginate.py and confirm paginate_slice handles limit <= 0.".into(),
                setup_files: BTreeMap::from([(
                    "app/paginate.py".into(),
                    "def paginate_slice(items: list[str], offset: int, limit: int) -> list[str]:\n    if limit <= 0:\n        return []\n    return items[max(0, offset):max(0, offset) + limit]\n".into(),
                )]),
                verifier_command: None,
                expected_substrings: vec!["paginate_slice".into()],
                domain_tags: vec!["python".into(), "needs_stale_hash_hint".into()],
            },
        ],
    }
}

fn go_preset() -> EvalSuiteSpec {
    EvalSuiteSpec {
        name: "allyh-go-preset".into(),
        domain: "go".into(),
        benchmark: "go-coherence-v1".into(),
        split_seed: 20_261_116,
        tasks: vec![
            EvalTask {
                id: "go-context-cancel".into(),
                title: "Propagate context cancellation cleanly".into(),
                prompt: "Inspect pkg/worker/worker.go and verify ProcessItem checks ctx.Err().".into(),
                setup_files: BTreeMap::from([(
                    "pkg/worker/worker.go".into(),
                    "package worker\n\nimport \"context\"\n\nfunc ProcessItem(ctx context.Context, id string) error {\n\tif err := ctx.Err(); err != nil {\n\t\treturn err\n\t}\n\t_ = id\n\treturn nil\n}\n".into(),
                )]),
                verifier_command: None,
                expected_substrings: vec!["ProcessItem".into()],
                domain_tags: vec!["go".into(), "needs_contract_guidelines".into()],
            },
            EvalTask {
                id: "go-error-wrap".into(),
                title: "Wrap errors with %w for errors.Is".into(),
                prompt: "Inspect pkg/store/store.go and verify LoadRecord wraps errors properly.".into(),
                setup_files: BTreeMap::from([(
                    "pkg/store/store.go".into(),
                    "package store\n\nimport \"fmt\"\n\nfunc LoadRecord(id string) (string, error) {\n\tif id == \"\" {\n\t\treturn \"\", fmt.Errorf(\"invalid record id\")\n\t}\n\treturn id, nil\n}\n".into(),
                )]),
                verifier_command: None,
                expected_substrings: vec!["LoadRecord".into()],
                domain_tags: vec!["go".into(), "needs_interlock".into()],
            },
            EvalTask {
                id: "go-slice-copy".into(),
                title: "Avoid aliasing caller slice".into(),
                prompt: "Inspect pkg/buf/buf.go and confirm CloneStrings returns an independent copy.".into(),
                setup_files: BTreeMap::from([(
                    "pkg/buf/buf.go".into(),
                    "package buf\n\nfunc CloneStrings(in []string) []string {\n\tout := make([]string, len(in))\n\tcopy(out, in)\n\treturn out\n}\n".into(),
                )]),
                verifier_command: None,
                expected_substrings: vec!["CloneStrings".into()],
                domain_tags: vec!["go".into(), "needs_verification_proof".into()],
            },
            EvalTask {
                id: "go-config-defaults".into(),
                title: "Apply default timeout when zero".into(),
                prompt: "Inspect pkg/cfg/cfg.go and verify EffectiveTimeoutSeconds falls back to 30.".into(),
                setup_files: BTreeMap::from([(
                    "pkg/cfg/cfg.go".into(),
                    "package cfg\n\nfunc EffectiveTimeoutSeconds(v int) int {\n\tif v <= 0 {\n\t\treturn 30\n\t}\n\treturn v\n}\n".into(),
                )]),
                verifier_command: None,
                expected_substrings: vec!["EffectiveTimeoutSeconds".into()],
                domain_tags: vec!["go".into(), "needs_stale_hash_hint".into()],
            },
        ],
    }
}

fn polyglot_combined_preset() -> EvalSuiteSpec {
    let mut tasks = Vec::new();
    tasks.extend(rust_preset().tasks.into_iter().take(2));
    tasks.extend(typescript_preset().tasks.into_iter().take(2));
    tasks.extend(python_preset().tasks.into_iter().take(2));
    tasks.extend(go_preset().tasks.into_iter().take(2));
    EvalSuiteSpec {
        name: "allyh-polyglot-preset".into(),
        domain: "polyglot".into(),
        benchmark: "polyglot-coherence-v1".into(),
        split_seed: 20_261_117,
        tasks,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_polyglot_presets_build_and_split_cleanly() {
        for &name in list_suite_presets() {
            let suite = polyglot_suite_preset(name).expect("preset should build");
            let (evolve, held_out) = suite.split().expect("preset should split cleanly");
            assert!(!evolve.is_empty(), "{name}: evolve split must be non-empty");
            assert!(!held_out.is_empty(), "{name}: held-out split must be non-empty");
            assert_eq!(evolve.len() + held_out.len(), suite.tasks.len());
        }
        assert!(polyglot_suite_preset("cobol").is_err());
    }
}

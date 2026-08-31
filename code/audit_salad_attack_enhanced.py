"""Dataset-only pairability audit for SALAD's attack-enhanced subset."""
import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "salad_attack_enhanced.parquet"
REPORT = ROOT / "Report" / "salad_attack_enhanced_pairability_audit.json"
SAMPLES = ROOT / "data" / "salad_attack_enhanced_semantic_samples.csv"


def norm(value: str) -> str:
    return " ".join(value.lower().split())


def pairs_different_groups(counts: list[int]) -> int:
    total = sum(counts)
    return (total * (total - 1) - sum(n * (n - 1) for n in counts)) // 2


def main() -> None:
    df = pd.read_parquet(DATA)
    required = {"qid", "baseq", "augq", "method"}
    assert required <= set(df), f"missing columns: {required - set(df)}"
    df = df.copy()
    df["base_norm"] = df.baseq.map(norm)
    df["aug_norm"] = df.augq.map(norm)
    df["base_verbatim_contained"] = [b in a for b, a in zip(df.base_norm, df.aug_norm)]

    qid_rows = df.groupby("qid").size()
    qid_methods = df.groupby("qid").method.nunique()
    base_identical = df.groupby("qid").base_norm.nunique()
    method_rows = df.method.value_counts().sort_index()

    positives = sum(
        pairs_different_groups(group.groupby("method").size().tolist())
        for _, group in df.groupby("qid")
    )
    negatives = sum(
        pairs_different_groups(group.groupby("qid").size().tolist())
        for _, group in df.groupby("method")
    )

    wrapper = {}
    for method, group in df.groupby("method", sort=True):
        contained = group[group.base_verbatim_contained]
        prefix, suffix = [], []
        for row in contained.itertuples():
            at = row.aug_norm.find(row.base_norm)
            prefix.append(at)
            suffix.append(len(row.aug_norm) - at - len(row.base_norm))
        wrapper[method] = {
            "rows": int(len(group)),
            "baseq_verbatim_contained_count": int(len(contained)),
            "baseq_verbatim_contained_fraction": float(len(contained) / len(group)),
            "rewrite_or_paraphrase_rate": float(1 - len(contained) / len(group)),
            "prefix_length_chars_when_contained": {
                "mean": float(pd.Series(prefix).mean()) if prefix else None,
                "median": float(pd.Series(prefix).median()) if prefix else None,
            },
            "suffix_length_chars_when_contained": {
                "mean": float(pd.Series(suffix).mean()) if suffix else None,
                "median": float(pd.Series(suffix).median()) if suffix else None,
            },
        }

    holdout = {}
    all_methods = sorted(method_rows.index)
    for method in all_methods:
        reduced = df[df.method != method]
        holdout[method] = {
            "remaining_methods": sorted(reduced.method.unique()),
            "qids_with_at_least_two_remaining_methods": int((reduced.groupby("qid").method.nunique() >= 2).sum()),
            "remaining_positive_pairs": int(sum(
                pairs_different_groups(g.groupby("method").size().tolist())
                for _, g in reduced.groupby("qid")
            )),
            "remaining_negative_pairs": int(sum(
                pairs_different_groups(g.groupby("qid").size().tolist())
                for _, g in reduced.groupby("method")
            )),
        }

    # This is the blinded review packet. It is not printed so audit output does not reproduce attack prompts.
    samples = pd.concat([group.sample(n=min(20, len(group)), random_state=20260826)
                         for _, group in df.groupby("method")], ignore_index=True)
    samples = samples.loc[:, ["qid", "method", "baseq", "augq", "base_verbatim_contained"]].assign(
        human_intent_label="", review_note=""
    )
    samples.to_csv(SAMPLES, index=False)

    report = {
        "dataset": "OpenSafetyLab/Salad-Data",
        "subset": "attack_enhanced_set/train",
        "source_file": str(DATA.relative_to(ROOT)),
        "columns": list(df.columns[:8]),
        "total_rows": int(len(df)),
        "unique_qids": int(df.qid.nunique()),
        "unique_methods": int(df.method.nunique()),
        "rows_per_method": {k: int(v) for k, v in method_rows.items()},
        "rows_per_qid": {"min": int(qid_rows.min()), "max": int(qid_rows.max()), "mean": float(qid_rows.mean())},
        "methods_per_qid": {str(k): int((qid_methods == k).sum()) for k in sorted(qid_methods.unique())},
        "qids_with_at_least_n_methods": {str(k): int((qid_methods >= k).sum()) for k in range(2, int(qid_methods.max()) + 1)},
        "baseq_identical_within_qid": {"all_qids": bool((base_identical == 1).all()), "violating_qids": int((base_identical != 1).sum())},
        "same_qid_different_method_positive_pairs": int(positives),
        "different_qid_same_method_negative_pairs": int(negatives),
        "whole_method_holdout_pairability": holdout,
        "mechanism_metrics": wrapper,
        "semantic_review": {
            "packet": str(SAMPLES.relative_to(ROOT)),
            "rows_per_method": 20,
            "status": "pending human intent-preservation labels; string containment is structural evidence only",
            "rejection_policy": "Reject only rows labeled CHANGED_INTENT or AMBIGUOUS by review; do not infer drift merely from a paraphrase."
        },
        "decision": "PENDING_SEMANTIC_REVIEW"
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["total_rows", "unique_qids", "unique_methods", "rows_per_method", "methods_per_qid", "qids_with_at_least_n_methods", "baseq_identical_within_qid", "same_qid_different_method_positive_pairs", "different_qid_same_method_negative_pairs", "decision"]}, indent=2))
    print(f"review_packet={SAMPLES.relative_to(ROOT)}")
    print(f"report={REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

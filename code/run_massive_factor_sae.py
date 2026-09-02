"""Run the locked MASSIVE factor-contrastive SAE configuration."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import intent_locale_relations as relations


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "code" / "run_massive_sparse_partition_pilot.py"
METADATA = ROOT / "data" / "massive_partition_artifacts" / "train_metadata.csv"
SEEDS = (20260827, 20260828, 20260829)
BASE_ARGS = (
    "--activation", "batchtopk",
    "--relation-sampler", "intent_50_50",
    "--c-fraction", "0.30",
    "--epochs", "30",
    "--anchors-per-epoch", "45968",
    "--batch-size", "128",
    "--eval-batch-size", "128",
    "--lr", "1e-4",
    "--temperature", "0.07",
    "--initialization", "fresh",
    "--swap-weight", "0",
    "--evaluation-split", "test",
)
EXPERIMENTS = {
    "batchtopk": ("--objective", "reconstruction", "--route-mode", "none", "--sparsifier", "global", "--contrast-weight", "0"),
    "block_control": ("--objective", "reconstruction", "--route-mode", "none", "--sparsifier", "block", "--contrast-weight", "0"),
    "matryoshka": ("--objective", "reconstruction", "--route-mode", "none", "--sparsifier", "global", "--sae-objective", "matryoshka", "--contrast-weight", "0"),
    "one_sided": ("--objective", "matched", "--route-mode", "c_only", "--sparsifier", "block", "--contrast-weight", "1"),
    "reciprocal": ("--objective", "matched", "--route-mode", "reciprocal", "--sparsifier", "block", "--contrast-weight", "1"),
    "triplet_m0p1": ("--objective", "triplet", "--triplet-margin", "0.1", "--route-mode", "reciprocal", "--sparsifier", "block", "--contrast-weight", "1"),
    "triplet_m0p2": ("--objective", "triplet", "--triplet-margin", "0.2", "--route-mode", "reciprocal", "--sparsifier", "block", "--contrast-weight", "1"),
    "triplet_m0p4": ("--objective", "triplet", "--triplet-margin", "0.4", "--route-mode", "reciprocal", "--sparsifier", "block", "--contrast-weight", "1"),
}


def audit_protocol():
    metadata = pd.read_csv(METADATA)
    index = relations.build_relation_index(metadata)
    rng = np.random.default_rng(SEEDS[0])
    rows = rng.choice(len(metadata), 10000, replace=False)
    c_positive, s_positive = relations.sample_intent_relations(rows, index, rng, 0.5)
    audit = relations.audit_sample(rows, c_positive, s_positive, index)
    assert audit["zC_exact_id_fraction"] == 0.5
    assert audit["zC_nonexact_different_id_fraction"] == 0.5
    assert audit["zC_same_intent_fraction"] == 1.0
    assert audit["zC_different_locale_fraction"] == 1.0
    assert audit["zS_same_locale_fraction"] == 1.0
    assert audit["zS_different_intent_fraction"] == 1.0
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="run all locked training seeds")
    parser.add_argument("--final", action="store_true", help="train the frozen 90/10 split for untouched-test evaluation")
    parser.add_argument("--validation", action="store_true", help="use one seed and a held-out validation split")
    parser.add_argument("--smoke", action="store_true", help="run one training step per selected experiment")
    parser.add_argument("--experiment", choices=tuple(EXPERIMENTS), action="append")
    parser.add_argument("--seed", type=int, choices=SEEDS, action="append")
    parser.add_argument("--width-multiplier", type=int, choices=(4, 8), default=4)
    args = parser.parse_args()
    assert sum(map(bool, (args.train, args.final, args.validation, args.smoke))) <= 1
    audit = audit_protocol()
    experiments = tuple(args.experiment or ("reciprocal",))
    seeds = tuple(args.seed or ((SEEDS[0],) if args.validation or args.smoke else SEEDS))
    assert not args.validation or len(seeds) == 1
    sparse_width, total_k, c_k = (9216, 64, 13) if args.width_multiplier == 4 else (18432, 128, 26)
    width_args = ("--sparse-width", str(sparse_width), "--total-k", str(total_k), "--c-k", str(c_k))
    label = "step5_width8" if args.width_multiplier == 8 else "step4_final" if args.final else "step2"
    protocol = {
        name: BASE_ARGS + width_args + EXPERIMENTS[name] + ("--label", f"{label}_{name}")
        for name in experiments
    }
    print(json.dumps({"seeds": seeds, "experiments": protocol, "relation_audit": audit}, indent=2))
    if args.train or args.final or args.validation or args.smoke:
        for name in experiments:
            extra = ()
            if args.final:
                extra = ("--validation-fraction", "0.1", "--evaluation-split", "test", "--skip-eval")
            if args.validation:
                extra = ("--validation-fraction", "0.1", "--evaluation-split", "validation")
            if args.smoke:
                extra = ("--epochs", "1", "--max-steps", "1", "--skip-eval")
            for seed in seeds:
                command = [sys.executable, str(TRAINER), *protocol[name], *extra, "--seed", str(seed)]
                subprocess.run(
                    command,
                    cwd=ROOT,
                    check=True,
                )


if __name__ == "__main__":
    main()

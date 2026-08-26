"""Write the requested dictionary-normalized Top-k SAE comparison table."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Report" / "topk_sae_sweep_report.json"
OUT = ROOT / "Report" / "TOPK_SAE_SWEEP.md"


def main():
    report = json.loads(SOURCE.read_text(encoding="utf-8")); lines = ["# Top-k SAE sparsity sweep", "", "What we did: trained matched Top-k SAEs on the frozen raw layer-8 and C-bottleneck activations. Both use the same 8,964 C-train rows, 747 facts, 4x expansion, seed, optimizer, and 30 epochs. Only k changes.", "", "Why: Top-k directly fixes the number of active features, avoiding an arbitrary L1 coefficient.", "", "| k | representation | reconstruction MSE | mean L0 | relation purity | domain purity | S-family purity | C-selective / dictionary | S-selective / dictionary |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k in report["k_values"]:
        for name, label in (("raw_gemma_layer8", "Raw Gemma"), ("c_bottleneck", "C bottleneck")):
            row = report[name]["by_k"][str(k)]; width = report[name]["feature_width"]
            c = row["C_relation_selective_features"]; s = row["S_family_selective_features"]
            lines.append(f"| {k} | {label} | {row['standardized_reconstruction_mse']:.3f} | {row['mean_L0']:.0f} | {row['mean_top50_C_relation_purity']:.3f} | {row['mean_top50_C_domain_purity']:.3f} | {row['mean_top50_S_family_purity']:.3f} | {c}/{width} ({c / width:.1%}) | {s}/{width} ({s / width:.1%}) |")
    lines += ["", "What we found: across every k, C-bottleneck features have much lower surface-family purity (about 0.28 versus 0.48–0.63 for raw Gemma) and a higher dictionary-normalized fraction of relation-selective features (12.6–17.8% versus 0.6–2.9%). Raw Gemma retains 0.3–1.2% S-selective features; the C bottleneck has 0% except 1 feature at k=32. Reconstruction improves as k grows, with no contrastive retraining.", ""]
    OUT.write_text("\n".join(lines), encoding="utf-8"); print(OUT.read_text(encoding="utf-8"))


if __name__ == "__main__": main()

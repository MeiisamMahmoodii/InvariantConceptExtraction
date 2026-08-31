"""Build the quantitative main-paper SVG figures from saved evaluation artifacts."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "paper" / "figure_data"
OUT = ROOT / "paper" / "figures"

RAW = "#7A7A7A"
BLUE = "#2878B5"
LIGHT_BLUE = "#9CC5E8"
ORANGE = "#E1812C"
LIGHT_ORANGE = "#F4BF8C"
GREEN = "#3A9D5D"
RED = "#C84C4C"
GRID = "#D8D8D8"


def load_json(name):
    with (DATA / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def intent_aucs(path):
    with (ROOT / path).open(encoding="utf-8", newline="") as handle:
        return {row["intent"]: float(row["heldout_auc"]) for row in csv.DictReader(handle)}


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
    })


def panel_label(ax, letter):
    ax.text(-0.16, 1.10, letter, transform=ax.transAxes, fontweight="bold", fontsize=14,
            va="top", ha="left")


def figure3():
    payload = load_json("figure3_quantitative.json")
    paired_path = ROOT / payload["paired_stability"]["per_id_csv"]
    with paired_path.open(encoding="utf-8", newline="") as handle:
        deltas = np.array([float(row["delta_matched_minus_infonce"]) for row in csv.DictReader(handle)])
    auc = payload["concept_auc"]
    raw_auc = intent_aucs(auc["raw_per_intent_csv"])
    matched_auc = intent_aucs(auc["matched_per_intent_csv"])
    assert len(deltas) == payload["paired_stability"]["n_semantic_ids"]
    assert set(raw_auc) == set(matched_auc) and len(raw_auc) == auc["n_intents"]

    fig = plt.figure(figsize=(13.4, 4.6), constrained_layout=True)
    axes = fig.subplot_mosaic([["A", "B", "C"]], width_ratios=[1.0, 1.05, 1.25])

    # A: direct, labelled stability comparison.
    ax = axes["A"]
    stability = payload["stability_summary"]
    labels = [row["representation"] for row in stability]
    values = [row["stability"] for row in stability]
    colors = [RAW, LIGHT_BLUE, BLUE, LIGHT_ORANGE, ORANGE]
    positions = np.arange(len(labels))[::-1]
    ax.hlines(positions, 0, values, color=GRID, lw=2, zorder=1)
    ax.scatter(values, positions, s=90, color=colors, zorder=2, edgecolor="white", linewidth=0.9)
    for x, y in zip(values, positions):
        ax.text(x + 0.012, y, f"{x:.3f}", va="center", fontsize=9)
    ax.set(yticks=positions, yticklabels=labels, xlim=(-0.02, 0.62), xlabel="Arabic–Chinese feature stability")
    ax.set_title("Held-out feature stability", loc="left", fontweight="bold", fontsize=12, pad=8)
    ax.grid(axis="x", color=GRID, lw=0.7)
    panel_label(ax, "A")

    # B: full paired semantic-ID distribution, not just an aggregate bar.
    ax = axes["B"]
    violin = ax.violinplot(deltas, positions=[0], vert=False, widths=0.72, showextrema=False)
    for body in violin["bodies"]:
        body.set_facecolor(BLUE)
        body.set_edgecolor(BLUE)
        body.set_alpha(0.18)
    rng = np.random.default_rng(0)
    shown = rng.choice(deltas, size=min(550, len(deltas)), replace=False)
    ax.scatter(shown, rng.normal(0, 0.055, len(shown)), s=14, color=BLUE,
               alpha=0.20, linewidths=0, zorder=2)
    ci_low, ci_high = payload["paired_stability"]["bootstrap_95_ci"]
    mean = payload["paired_stability"]["mean_difference"]
    ax.errorbar(mean, 0, xerr=[[mean - ci_low], [ci_high - mean]], fmt="D", color=GREEN,
                markersize=8, capsize=4, lw=2.1, zorder=4)
    ax.axvline(0, color=RED, alpha=0.55, lw=0.9, ls="--")
    ax.set(yticks=[], ylim=(-0.5, 0.5), xlim=(-0.55, 0.85),
           xlabel="Per-ID stability difference (Matched zC − InfoNCE zC)")
    ax.set_title("Paired stability gain by semantic ID", loc="left", fontweight="bold", fontsize=12, pad=8)
    ax.text(0.02, 0.98,
            f"mean +{mean:.3f}  |  95% CI [{ci_low:.3f}, {ci_high:.3f}]\n"
            f"{payload['paired_stability']['fraction_positive']:.1%} IDs improve  |  n = {len(deltas):,}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9.5, color=RAW)
    ax.grid(axis="x", color=GRID, lw=0.7)
    panel_label(ax, "B")

    # C: each intent's raw-to-matched move; lines make the breadth of effect visible.
    ax = axes["C"]
    pairs = sorted(((raw_auc[key], matched_auc[key]) for key in raw_auc), key=lambda pair: pair[1] - pair[0])
    for raw, matched in pairs:
        ax.plot([0, 1], [raw, matched], color=GREEN if matched >= raw else RED,
                alpha=0.18 if matched >= raw else 0.15, lw=0.85)
    ax.scatter(np.zeros(len(pairs)), [pair[0] for pair in pairs], color=RAW, alpha=0.65, s=11, zorder=3)
    ax.scatter(np.ones(len(pairs)), [pair[1] for pair in pairs], color=BLUE, alpha=0.65, s=11, zorder=3)
    ax.scatter([0, 1], [np.mean([pair[0] for pair in pairs]), np.mean([pair[1] for pair in pairs])],
               color=[RAW, BLUE], marker="D", s=58, zorder=4, edgecolor="white", linewidth=0.8)
    ax.set(xlim=(-0.28, 1.28), xticks=[0, 1], xticklabels=["Raw H", "Matched zC"], ylim=(0.62, 1.01),
           ylabel="Held-out intent AUC")
    ax.set_title("Concept quality across intents", loc="left", fontweight="bold", fontsize=12, pad=8)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.text(0.03, 0.06, f"mean Δ = +{auc['mean_difference']:.3f}  |  95% CI [{auc['bootstrap_95_ci'][0]:.3f}, {auc['bootstrap_95_ci'][1]:.3f}]\n"
            f"{sum(matched > raw for raw, matched in pairs)}/{len(pairs)} intents improve ({auc['fraction_concepts_improved']:.1%})",
            transform=ax.transAxes, fontsize=8.8,
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": GRID, "lw": 0.8})
    panel_label(ax, "C")
    fig.savefig(OUT / "figure3_quantitative.svg", format="svg", bbox_inches="tight")
    fig.savefig(OUT / "figure3_quantitative.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)


def figure4():
    payload = load_json("figure4_interventions.json")
    fig, ax = plt.subplots(figsize=(3.35, 3.15), constrained_layout=True)

    # Intended factor transfer versus cross-factor leakage/control.
    rows = payload["specificity_matrix"]
    columns = [("donor_intent", "Donor intent\nfollows"), ("donor_locale", "Donor locale\nfollows"),
               ("unrelated_control", "Unrelated\ncontrol")]
    ax.set(xlim=(-0.5, 2.5), ylim=(1.5, -0.5), xticks=range(3), xticklabels=[label for _, label in columns],
           yticks=range(2), yticklabels=["zC swap", "zS swap"])
    for row_index, row in enumerate(rows):
        for col_index, (key, _) in enumerate(columns):
            value = row[key]
            intended = (row["intervention"] == "zC swap" and key == "donor_intent") or (row["intervention"] == "zS swap" and key == "donor_locale")
            color = GREEN if intended else RED
            alpha = 0.18 + 0.70 * value if intended else 0.10 + 0.18 * min(value, 0.12)
            ax.add_patch(plt.Rectangle((col_index - 0.45, row_index - 0.39), 0.90, 0.78,
                                       facecolor=color, alpha=alpha, edgecolor="none"))
            ax.text(col_index, row_index - 0.06, f"{value:.1%}", ha="center", va="center",
                    fontsize=11.5, fontweight="bold")
            ax.text(col_index, row_index + 0.23,
                    "intended transfer" if intended else ("cross-factor\nleakage" if key != "unrelated_control" else "label control"),
                    ha="center", va="center", fontsize=6.2)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="x", labelsize=7.3, pad=5)
    ax.tick_params(axis="y", labelsize=7.8, pad=4)
    ax.set_title("Representation-swap specificity", loc="left", fontweight="bold", fontsize=10)
    ax.text(0.0, -0.19, f"{payload['specificity_metadata']['pairs']:,} matched cases; controls use unrelated labels.",
            transform=ax.transAxes, fontsize=6.6)
    fig.savefig(OUT / "figure4_interventions.svg", format="svg", bbox_inches="tight")
    fig.savefig(OUT / "figure4_interventions.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)


def figure4_generation_appendix():
    payload = load_json("figure4_interventions.json")
    fig, ax = plt.subplots(figsize=(7.0, 4.7), constrained_layout=True)
    # Three generation metrics with Wilson intervals for the scale-200 S-steering run.
    conditions = payload["generation_steering"]
    metrics = [("target_language_success", "Target-language\nsuccess"),
               ("source_intent_preserved", "Source-intent\npreserved"),
               ("donor_intent_leakage", "Donor-intent\nleakage")]
    x = np.arange(len(metrics))
    width = 0.23
    colors = [RAW, ORANGE, LIGHT_ORANGE]
    labels = ["No intervention", "True zS swap (α = .75)", "Random zS control"]
    for i, (condition, color, label) in enumerate(zip(conditions, colors, labels)):
        values = [condition[key]["rate"] for key, _ in metrics]
        lower = [value - condition[key]["wilson_95_ci"][0] for value, (key, _) in zip(values, metrics)]
        upper = [condition[key]["wilson_95_ci"][1] - value for value, (key, _) in zip(values, metrics)]
        bars = ax.bar(x + (i - 1) * width, values, width, color=color, label=label, zorder=2,
                      hatch="//" if i == 2 else None, edgecolor="white" if i == 2 else "none", linewidth=0.5)
        ax.errorbar(x + (i - 1) * width, values, yerr=[lower, upper], fmt="none", color="#404040", capsize=2.5, lw=1, zorder=3)
        for bar, value in zip(bars, values):
            if value >= 0.10:
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.035, f"{value:.1%}", ha="center", va="bottom", fontsize=8)
    ax.set(xticks=x, xticklabels=[label for _, label in metrics], ylim=(0, 0.46), ylabel="Rate (Wilson 95% CI)")
    ax.set_title("Generation-level S steering is partial", loc="left", fontweight="bold")
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.text(0.01, -0.23, "Representation-level swapping is highly specific; open-loop generation steering is partial.",
            transform=ax.transAxes, fontsize=8.5)
    panel_label(ax, "A")
    fig.savefig(OUT / "figure4_generation_appendix.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    figure3()
    figure4()
    figure4_generation_appendix()
    assert (OUT / "figure3_quantitative.svg").exists()
    assert (OUT / "figure4_interventions.svg").exists()
    assert (OUT / "figure4_generation_appendix.svg").exists()

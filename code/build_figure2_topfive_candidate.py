"""Build an alternative Figure 2 with five independently selected SAE features per dictionary."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "paper" / "figure_data" / "figure2_topfive_features.json"
METRICS = ROOT / "paper" / "figure_data" / "figure2_features.json"
OUT = ROOT / "paper" / "figures" / "figure2_topfive_candidate.svg"

GRAY, LIGHT_GRAY = "#727982", "#EFF2F4"
BLUE, LIGHT_BLUE = "#2878B5", "#E7F1FA"
ORANGE, LIGHT_ORANGE = "#E1812C", "#FFF0E1"
GREEN, TEXT = "#3A9D5D", "#20252B"


def card(ax, y, item, title_color, face):
    ax.add_patch(FancyBboxPatch((.07, y), .86, .095, boxstyle="round,pad=.014,rounding_size=.018",
                                transform=ax.transAxes, facecolor=face, edgecolor=title_color, linewidth=.9))
    orientation = "intent" if item["orientation"] == "intent" else "language"
    ax.text(.11, y + .062, f"Feature {item['feature_id']}  ·  {orientation}", transform=ax.transAxes,
            ha="left", va="center", fontsize=8.8, color=title_color, fontweight="bold")
    ax.text(.11, y + .030, f"{item['label']}  —  {item['evidence']}", transform=ax.transAxes,
            ha="left", va="center", fontsize=7.35, color=TEXT)


def orientation_bar(ax, metrics, representation):
    values = (metrics["intent_oriented"], metrics["language_oriented"], metrics["mixed_other"])
    colors = (BLUE, ORANGE, LIGHT_GRAY)
    left = .07
    for value, color in zip(values, colors):
        ax.add_patch(FancyBboxPatch((left, .115), .86 * value, .042, boxstyle="round,pad=0,rounding_size=.01",
                                    transform=ax.transAxes, facecolor=color, edgecolor="none"))
        left += .86 * value
    labels = (f"intent {100 * values[0]:.1f}%", f"language {100 * values[1]:.1f}%", f"mixed {100 * values[2]:.1f}%")
    ax.text(.07, .175, "  ·  ".join(labels), transform=ax.transAxes, ha="left", va="bottom", fontsize=7.25, color=GRAY)
    stability = "≈ 0.000" if abs(metrics["heldout_stability"]) < .005 else f"{metrics['heldout_stability']:.3f}"
    ax.text(.50, .055, f"Cross-language stability: {stability}", transform=ax.transAxes,
            ha="center", va="center", fontsize=8.6, color=GREEN if representation == "zC" else GRAY,
            fontweight="bold" if representation == "zC" else "normal")


def column(ax, title, subtitle, items, metrics, accent, face, representation):
    ax.set_axis_off()
    ax.text(.50, .97, title, transform=ax.transAxes, ha="center", va="top", fontsize=14, color=accent, fontweight="bold")
    ax.text(.50, .925, subtitle, transform=ax.transAxes, ha="center", va="top", fontsize=8.7, color=GRAY)
    ax.text(.07, .855, "Five highest-selectivity criterion-passing features", transform=ax.transAxes,
            ha="left", va="center", fontsize=7.4, color=GRAY, style="italic")
    for y, item in zip((.74, .61, .48, .35, .22), items):
        card(ax, y, item, BLUE if item["orientation"] == "intent" else ORANGE, face if item["orientation"] == "intent" else LIGHT_ORANGE)
    orientation_bar(ax, metrics, representation)


def main():
    with FEATURES.open(encoding="utf-8") as handle:
        features = json.load(handle)
    with METRICS.open(encoding="utf-8") as handle:
        metrics = {row["representation"]: row for row in json.load(handle)["aggregate_metrics"]}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "path"})
    fig, axes = plt.subplots(1, 3, figsize=(14.3, 6.2))
    column(axes[0], "Raw H", "mixed SAE dictionary", features["H"], metrics["H"], GRAY, LIGHT_GRAY, "H")
    column(axes[1], "zC", "preserved-dominant dictionary", features["zC"], metrics["zC"], BLUE, LIGHT_BLUE, "zC")
    column(axes[2], "zS", "varying-dominant dictionary", features["zS"], metrics["zS"], ORANGE, LIGHT_ORANGE, "zS")
    fig.text(.50, .988, "Candidate Figure 2 — Top-five SAE feature evidence", ha="center", va="top", fontsize=15, fontweight="bold", color=TEXT)
    fig.subplots_adjust(left=.035, right=.985, bottom=.06, top=.91, wspace=.12)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, format="svg")
    plt.close(fig)
    assert OUT.exists()


if __name__ == "__main__":
    main()

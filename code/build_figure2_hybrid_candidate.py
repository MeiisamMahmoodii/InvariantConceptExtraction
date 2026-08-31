"""Build a third Figure 2 candidate: composition bars plus exemplar callouts."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "paper" / "figure_data" / "figure2_features.json"
OUT = ROOT / "paper" / "figures" / "figure2_hybrid_candidate.svg"

GRAY, LIGHT_GRAY = "#727982", "#DDE2E6"
BLUE, LIGHT_BLUE = "#2878B5", "#E7F1FA"
ORANGE, LIGHT_ORANGE = "#E1812C", "#FFF0E1"
GREEN, TEXT, GRID = "#3A9D5D", "#20252B", "#DCE2E6"


def exemplar(fig, y, title, subtitle, detail, edge, face):
    width, height = .86, .075
    fig.add_artist(FancyBboxPatch((.5 - width / 2, y), width, height,
                                boxstyle="round,pad=.015,rounding_size=.025",
                                transform=fig.transFigure, clip_on=False,
                                facecolor=face, edgecolor=edge, linewidth=1.1))
    fig.text(.5, y + .058, title, ha="center", va="center",
             fontsize=7.7, color=edge, fontweight="bold")
    fig.text(.5, y + .036, subtitle, ha="center", va="center",
             fontsize=7.0, color=TEXT, fontweight="bold")
    fig.text(.5, y + .014, detail, ha="center", va="center",
             fontsize=6.6, color=GRAY)


def main():
    with DATA.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    metrics = {row["representation"]: row for row in payload["aggregate_metrics"]}
    examples = {row["representation"] + str(row["feature_id"]): row for row in payload["exemplar_features"]}

    names = ["Raw H", "zC", "zS"]
    keys = ["H", "zC", "zS"]
    intent = [100 * metrics[key]["intent_oriented"] for key in keys]
    language = [100 * metrics[key]["language_oriented"] for key in keys]
    mixed = [100 * metrics[key]["mixed_other"] for key in keys]
    stability = [metrics[key]["heldout_stability"] for key in keys]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "path"})
    fig = plt.figure(figsize=(3.35, 5.45))
    ax = fig.add_axes([.22, .52, .72, .25])
    y = range(3)
    height = .55
    ax.barh(y, intent, height, color=BLUE, label="Intent-oriented", zorder=3)
    ax.barh(y, language, height, left=intent, color=ORANGE, label="Language-oriented", zorder=3)
    ax.barh(y, mixed, height, left=[a + b for a, b in zip(intent, language)], color=LIGHT_GRAY,
            label="Mixed / other", zorder=3)

    for index in y:
        segments = [(intent[index], 0, "intent"), (language[index], intent[index], "language"),
                    (mixed[index], intent[index] + language[index], "mixed / other")]
        for value, left, label in segments:
            if value >= 10:
                color = TEXT if label == "mixed / other" else "white"
                ax.text(left + value / 2, index, f"{label}\n{value:.1f}%", ha="center", va="center",
                        fontsize=7.3, color=color, fontweight="bold", linespacing=1.0)
        if mixed[index] > 0:
            ax.text(100, index - .38, f"mixed / other {mixed[index]:.1f}%", ha="right", va="center",
                    fontsize=6.7, color=TEXT, fontweight="bold", clip_on=False)
        shown = "≈ 0.000" if abs(stability[index]) < .005 else f"{stability[index]:.3f}"
        ax.text(0, index + .40, f"Cross-language stability  {shown}", ha="left", va="center",
                fontsize=8.7, color=GREEN if index == 1 else GRAY,
                fontweight="bold" if index == 1 else "normal", clip_on=False)

    ax.set_xlim(0, 100)
    ax.set_ylim(-.5, 2.6)
    ax.invert_yaxis()
    ax.set_yticks(list(y), names, fontsize=9.2, fontweight="bold")
    ax.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100"])
    ax.set_xlabel("Share of active SAE features (%)", fontsize=8.3, labelpad=4)
    ax.grid(axis="x", color=GRID, linewidth=.75, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=7.6)
    raw = examples["H7124"]
    zc = examples["zC334"]
    zs = examples["zS491"]
    exemplar(fig, .315, "Raw H · feature 7124", "intent feature in a mixed dictionary",
             "JOKE · Spanish / English / Hindi", GRAY, "#F3F5F6")
    exemplar(fig, .205, "zC · feature 334", "same intent across different languages",
             "COFFEE · Thai / Portuguese / Romanian", BLUE, LIGHT_BLUE)
    exemplar(fig, .095, "zS · feature 491", "same language across different intents",
             "AMHARIC · calendar / email / entertainment", ORANGE, LIGHT_ORANGE)

    fig.text(.5, .975, "Matched conditioning reorganizes\nconcept features",
             ha="center", va="top", fontsize=10.5, color=TEXT, fontweight="bold", linespacing=1.15)
    fig.text(.5, .900, "Composition of each complete active SAE dictionary",
             ha="center", va="top", fontsize=7.4, color=GRAY)
    fig.text(.5, .860, "Representative features selected independently within each dictionary.",
             ha="center", va="top", fontsize=6.8, color=GRAY, style="italic")
    fig.text(.5, .025, "Raw is mixed; zC concentrates preserved-factor features;\nzS concentrates varying-factor features.",
             ha="center", va="bottom", fontsize=7.2, color=TEXT,
             fontweight="bold", linespacing=1.25)
    fig.savefig(OUT, format="svg", bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)
    assert OUT.exists()


if __name__ == "__main__":
    main()

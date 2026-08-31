"""Build Figure 1: upstream conditioning before unchanged concept extraction."""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"

GRAY = "#7A7A7A"
LIGHT_GRAY = "#EEF1F4"
BLUE = "#2878B5"
LIGHT_BLUE = "#E7F1FA"
ORANGE = "#E1812C"
LIGHT_ORANGE = "#FFF0E1"
GREEN = "#3A9D5D"
PURPLE = "#7A5AA6"
BORDER = "#AAB3BA"
TEXT = "#20252B"


def box(ax, x, y, w, h, title, subtitle="", face="white", edge=BORDER, title_color=TEXT, fontsize=10, lw=1.15):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.025,rounding_size=0.12",
                                linewidth=lw, facecolor=face, edgecolor=edge, zorder=3))
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", color=title_color,
            fontsize=fontsize, fontweight="bold", zorder=4)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.31, subtitle, ha="center", va="center", color=TEXT,
                fontsize=fontsize - 1.3, linespacing=1.25, zorder=4)


def arrow(ax, start, end, color=GRAY, rad=0.0, dashed=False, width=1.45):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, linewidth=width,
                                linestyle="--" if dashed else "-", color=color,
                                connectionstyle=f"arc3,rad={rad}", zorder=1))


def pair_card(ax, x, y, c_label, s_label):
    ax.add_patch(FancyBboxPatch((x, y), 1.23, 0.97, boxstyle="round,pad=0.02,rounding_size=0.09",
                                linewidth=0.95, facecolor="white", edgecolor=BORDER, zorder=3))
    ax.text(x + 0.615, y + 0.76, "paired example", ha="center", va="center", fontsize=7.3, color=GRAY, zorder=4)
    box(ax, x + 0.13, y + 0.22, 0.43, 0.29, c_label, face=LIGHT_BLUE, edge=BLUE, title_color=BLUE, fontsize=8.2, lw=0.8)
    box(ax, x + 0.67, y + 0.22, 0.43, 0.29, s_label, face=LIGHT_ORANGE, edge=ORANGE, title_color=ORANGE, fontsize=8.2, lw=0.8)


def latent_box(ax, x, y, title, role, detail, face, edge):
    ax.add_patch(FancyBboxPatch((x, y), 2.25, 1.42, boxstyle="round,pad=0.025,rounding_size=0.12",
                                linewidth=1.15, facecolor=face, edgecolor=edge, zorder=3))
    ax.text(x + 1.125, y + 1.03, title, ha="center", va="center", color=edge, fontsize=13.2,
            fontweight="bold", zorder=4)
    ax.text(x + 1.125, y + 0.69, role, ha="center", va="center", color=edge, fontsize=9.6,
            fontweight="bold", zorder=4)
    ax.text(x + 1.125, y + 0.31, detail, ha="center", va="center", color=TEXT, fontsize=8.3,
            linespacing=1.25, zorder=4)


def main():
    sns.set_theme(style="white", font="DejaVu Sans")
    # Paths prevent viewer-specific font substitution from shifting labels in the SVG.
    plt.rcParams.update({"svg.fonttype": "path", "font.size": 10})
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16.6, 6.0))
    ax.set(xlim=(0, 16.5), ylim=(0, 8.3))
    ax.axis("off")

    # Deliberate visual separation between the four pipeline stages.
    for x in (3.68, 6.86, 10.10):
        ax.plot([x, x], [1.20, 7.58], color="#D7DDE2", lw=0.9, ls=(0, (2, 4)), zorder=0)

    # Stage headers deliberately sit above all arrows and cards.
    headers = [(1.84, "1.  Controlled matched contrasts"), (5.27, "2.  Frozen representation"),
               (8.48, "3.  Upstream partition"), (13.30, "4.  Existing extractors and analyses")]
    for x, label in headers:
        ax.text(x, 7.97, label, ha="center", va="center", fontsize=11.3, fontweight="bold", color=TEXT)

    # 1. Exact controlled pair constructions.
    pair_card(ax, 0.35, 5.40, r"$C_i$", r"$S_a$")
    pair_card(ax, 1.85, 5.40, r"$C_i$", r"$S_b$")
    ax.text(1.35, 6.64, "same C  •  different S", ha="center", fontsize=8.8, color=BLUE, fontweight="bold")
    ax.text(1.35, 5.03, r"positive for $z_C$: keep close", ha="center", fontsize=8.7, color=BLUE)
    ax.text(1.35, 4.77, "contrast: different C, same S", ha="center", fontsize=6.8, color=GRAY)
    pair_card(ax, 0.35, 2.15, r"$C_i$", r"$S_a$")
    pair_card(ax, 1.85, 2.15, r"$C_j$", r"$S_a$")
    ax.text(1.35, 3.39, "different C  •  same S", ha="center", fontsize=8.8, color=ORANGE, fontweight="bold")
    ax.text(1.35, 1.78, r"positive for $z_S$: keep close", ha="center", fontsize=8.7, color=ORANGE)
    ax.text(1.35, 1.52, "contrast: same C, different S", ha="center", fontsize=6.8, color=GRAY)

    # 2. Frozen pooled representation used by the main MASSIVE pipeline.
    box(ax, 4.10, 3.68, 2.32, 1.52, "Frozen Gemma activation H",
        "layer-8 pooled representation\nH ∈ R²³⁰⁴", face=LIGHT_GRAY, edge=GRAY, fontsize=7.7)
    ax.text(5.26, 3.74, "frozen activations only • no fine-tuning", ha="center", va="center", fontsize=6.8, color=GRAY, zorder=4)
    arrow(ax, (3.16, 5.86), (4.10, 4.70), color=BLUE, rad=-0.10)
    arrow(ax, (3.16, 2.63), (4.10, 4.17), color=ORANGE, rad=0.10)

    # 3. The paired representation partition. Text and residual connector are kept inside this stage.
    latent_box(ax, 7.38, 5.12, "zC", "preserved-dominant", "128 dimensions\npreserved-factor route", LIGHT_BLUE, BLUE)
    latent_box(ax, 7.38, 2.14, "zS", "varying-dominant", "128 dimensions\nvarying-factor route", LIGHT_ORANGE, ORANGE)
    arrow(ax, (6.42, 4.78), (7.38, 5.82), color=BLUE, rad=-0.10)
    arrow(ax, (6.42, 4.10), (7.38, 2.86), color=ORANGE, rad=0.10)
    arrow(ax, (8.50, 5.12), (8.50, 3.56), color="#B7A4D0", rad=0.38, dashed=True, width=0.8)
    ax.text(8.68, 4.30, "residual dependence\nmay remain", ha="left", va="center", fontsize=7.25, color="#A48AC3")
    ax.text(8.50, 1.62, "relative specialization\nnot perfect disentanglement", ha="center", va="center",
            fontsize=7.25, color=GRAY, linespacing=1.25)

    # 4. Unchanged downstream machinery is a grouped region, so arrows never cross its contents.
    ax.add_patch(FancyBboxPatch((10.35, 1.42), 5.72, 5.15, boxstyle="round,pad=0.03,rounding_size=0.15",
                                linewidth=1.0, linestyle="--", facecolor="#FAFBFC", edgecolor=BORDER, zorder=2))
    ax.text(10.64, 6.21, "evaluated extractor objectives unchanged", ha="left", va="center", fontsize=9, color=GRAY, fontweight="bold", zorder=4)
    arrow(ax, (9.63, 5.84), (10.35, 5.84), color=BLUE)
    arrow(ax, (9.63, 2.86), (10.35, 2.86), color=ORANGE)
    box(ax, 10.72, 4.75, 2.20, 0.88, "Matched Top-k SAE", "dictionary learning", face="white", edge=BLUE, title_color=BLUE, fontsize=9.2)
    box(ax, 13.38, 4.75, 2.20, 0.88, "ConCA-style analysis", "concept-coherence analysis", face="white", edge=BLUE, title_color=BLUE, fontsize=9.0)
    box(ax, 10.72, 3.28, 2.20, 0.88, "Frozen Gemma Scope", "token-level audit", face="white", edge=PURPLE, title_color=PURPLE, fontsize=9.1)
    box(ax, 13.38, 3.22, 2.20, 1.00, "Factor-specific\nlatent swaps", "representation-level", face="white", edge=ORANGE, title_color=ORANGE, fontsize=8.4)
    box(ax, 10.72, 1.78, 4.86, 0.95, "Relative specialization, not perfect disentanglement",
        "zC: more stable preserved-factor concepts\nzS: varying-factor features become more concentrated",
        face="#F6FAF7", edge=GREEN, title_color=GREEN, fontsize=8.9)
    ax.text(13.22, 0.78, "Partition first; reuse the evaluated extractors.", ha="center", va="center",
            fontsize=8.4, color=GREEN, fontweight="bold")

    fig.savefig(OUT / "figure1_method_overview.svg", format="svg", bbox_inches="tight")
    fig.savefig(OUT / "figure1_method_overview.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)
    assert (OUT / "figure1_method_overview.svg").exists()


if __name__ == "__main__":
    main()

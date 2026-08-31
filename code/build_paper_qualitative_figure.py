"""Build Figure 2: problem → matched partition → specialized SAE features."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
TRACE_DATA = ROOT / "paper" / "figure_data" / "figure2_feature_traces.json"
OUT = ROOT / "paper" / "figures"

GRAY, LIGHT_GRAY = "#727982", "#EFF2F4"
BLUE, LIGHT_BLUE = "#2878B5", "#E7F1FA"
ORANGE, LIGHT_ORANGE = "#E1812C", "#FFF0E1"
GREEN, TEXT, BORDER = "#3A9D5D", "#20252B", "#B7C0C8"
LOCALE_COLORS = ["#5A91E3", "#77A9C7", "#77A9C7", "#D09052", "#9A78C0", "#5A91E3", "#D09052", "#6BA078"]
INTENT_COLORS = ["#5A91E3", "#5A91E3", "#5A91E3", "#5A91E3", "#D09052", "#78A96B", "#9A78C0", "#D76C6C"]


def box(ax, x, y, w, h, title, subtitle="", face="white", edge=BORDER, title_color=TEXT, size=10, pad=.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={pad},rounding_size=0.035",
                                transform=ax.transAxes, facecolor=face, edgecolor=edge, linewidth=1.05, zorder=3))
    ax.text(x + w / 2, y + h * .62, title, transform=ax.transAxes, ha="center", va="center", fontsize=size,
            color=title_color, fontweight="bold", zorder=4)
    if subtitle:
        ax.text(x + w / 2, y + h * .30, subtitle, transform=ax.transAxes, ha="center", va="center", fontsize=size - 1.9,
                color=TEXT, linespacing=1.15, zorder=4)


def arrow(ax, start, end, color=GRAY, rad=0, width=1.2, dashed=False):
    ax.add_patch(FancyArrowPatch(start, end, transform=ax.transAxes, arrowstyle="-|>", mutation_scale=10,
                                linewidth=width, color=color, linestyle="--" if dashed else "-",
                                connectionstyle=f"arc3,rad={rad}", zorder=1))


def panel_title(ax, letter, title, subtitle):
    ax.text(.01, .98, letter, transform=ax.transAxes, va="top", ha="left", fontsize=14, fontweight="bold", color=TEXT)
    ax.text(.08, .98, title, transform=ax.transAxes, va="top", ha="left", fontsize=11.4, fontweight="bold", color=TEXT)
    ax.text(.08, .84 if "\n" in title else .90, subtitle, transform=ax.transAxes, va="top", ha="left", fontsize=8.6, color=GRAY)


def panel_a(ax):
    panel_title(ax, "A", "Raw activations mix preserved and varying factors", "One pooled representation H carries both controlled relations.")
    ax.text(.05, .77, "same intent · multiple locales", transform=ax.transAxes, color=BLUE, fontsize=8.7, fontweight="bold")
    for x, locale in zip((.06, .19, .32), ("es", "en", "hi")):
        box(ax, x, .66, .10, .075, locale, "JOKE", face=LIGHT_BLUE, edge=BLUE, title_color=BLUE, size=7.6, pad=.005)
        arrow(ax, (x + .05, .66), (.49, .47), color=BLUE, rad=-.10, width=.85)
    ax.text(.05, .13, "same locale · multiple intents", transform=ax.transAxes, color=ORANGE, fontsize=8.7, fontweight="bold")
    for x, intent in zip((.06, .19, .32), ("calendar", "email", "music")):
        box(ax, x, .19, .10, .075, "am", intent, face=LIGHT_ORANGE, edge=ORANGE, title_color=ORANGE, size=6.6, pad=.005)
        arrow(ax, (x + .05, .265), (.49, .47), color=ORANGE, rad=.10, width=.85)
    box(ax, .48, .34, .38, .28, "", "", face=LIGHT_GRAY, edge=GRAY, title_color=TEXT, size=12)
    ax.text(.67, .565, "Raw H", transform=ax.transAxes, ha="center", va="center", fontsize=12, color=TEXT, fontweight="bold", zorder=5)
    for y, color, label in ((.490, BLUE, "intent cues"), (.410, ORANGE, "locale cues")):
        ax.add_patch(FancyBboxPatch((.55, y), .24, .038, boxstyle="round,pad=0.01,rounding_size=.02",
                                    transform=ax.transAxes, facecolor=color, alpha=.63, edgecolor="none", zorder=4))
        ax.text(.67, y + .019, label, transform=ax.transAxes, ha="center", va="center", fontsize=6.5, color="white", zorder=5)
    ax.text(.67, .370, "mixed pooled activation", transform=ax.transAxes, ha="center", va="center", fontsize=7.0, color=GRAY, zorder=5)
    ax.text(.67, .10, "Extractor sees both factor types at once", transform=ax.transAxes, ha="center", va="center",
            fontsize=9.0, color=GRAY, fontweight="bold")


def mini_pair(ax, x, y, labels, color, caption):
    for offset, label in zip((0, .115), labels):
        box(ax, x + offset, y, .095, .065, label[0], label[1], face=LIGHT_BLUE if color == BLUE else LIGHT_ORANGE,
            edge=color, title_color=color, size=6.9, pad=.005)
    ax.text(x + .105, y - .040, caption, transform=ax.transAxes, ha="center", va="center", fontsize=7.0,
            color=color, fontweight="bold")


def panel_b(ax):
    panel_title(ax, "B", "Matched contrasts induce relative specialization", "Partition first; preserve the opposite controlled relation as contrast.")
    box(ax, .07, .39, .20, .17, "H", "frozen pooled\nactivation", face=LIGHT_GRAY, edge=GRAY, title_color=TEXT, size=12)
    mini_pair(ax, .32, .74, (("Cᵢ", "Sₐ"), ("Cᵢ", "Sᵦ")), BLUE, "same C · different S")
    mini_pair(ax, .32, .21, (("Cᵢ", "Sₐ"), ("Cⱼ", "Sₐ")), ORANGE, "different C · same S")
    box(ax, .68, .61, .26, .24, "zC", "preserved-dominant\ncontent / invariant", face=LIGHT_BLUE, edge=BLUE, title_color=BLUE, size=10)
    box(ax, .68, .14, .26, .24, "zS", "varying-dominant\nsurface / locale", face=LIGHT_ORANGE, edge=ORANGE, title_color=ORANGE, size=10)
    arrow(ax, (.27, .50), (.665, .76), color=BLUE, rad=-.10)
    arrow(ax, (.27, .44), (.665, .26), color=ORANGE, rad=.10)
    arrow(ax, (.54, .775), (.665, .82), color=BLUE, rad=.04, width=.8, dashed=True)
    arrow(ax, (.54, .242), (.665, .17), color=ORANGE, rad=-.04, width=.8, dashed=True)
    ax.text(.50, .07, "zC pulls together same-C pairs  •  zS pulls together same-S pairs", transform=ax.transAxes,
            ha="center", va="center", fontsize=8.35, color=GREEN, fontweight="bold")


def trace(ax, rows, representation, feature, title, colors, labels, accent, y_max):
    values = [row["activation"] for row in rows]
    x = range(1, len(values) + 1)
    bars = ax.bar(x, values, color=colors, width=.70, zorder=3)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + y_max * .018, f"{value:.2f}", ha="center", va="bottom", fontsize=6.4)
    ax.text(.00, 1.08, f"{representation} feature {feature} · {title}", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=8.8, color=accent, fontweight="bold", clip_on=False)
    ax.set(ylim=(0, y_max), xlim=(.35, 8.65), xticks=list(x), xticklabels=labels, ylabel="activation")
    ax.tick_params(axis="x", labelsize=6.5, pad=1)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.grid(axis="y", color="#DEE4E8", lw=.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)


def panel_c(ax, traces):
    panel_title(ax, "C", "Matched partitioning changes\nwhat sparse features represent", "Top activations expose a semantic feature in zC and a language feature in zS.")
    raw = ax.inset_axes([.10, .61, .84, .13])
    top = ax.inset_axes([.10, .36, .84, .13])
    bottom = ax.inset_axes([.10, .13, .84, .13])
    h = traces["H_7124"]
    zc = traces["z_C_334"]
    zs = traces["z_S_491"]
    trace(raw, h, "Raw H", 7124, "JOKE intent", LOCALE_COLORS,
          [row["locale"].split("-")[0] for row in h], GRAY, 42.0)
    trace(top, zc, "zC", 334, "COFFEE intent", LOCALE_COLORS,
          [row["locale"].split("-")[0] for row in zc], BLUE, 7.6)
    trace(bottom, zs, "zS", 491, "AMHARIC language", INTENT_COLORS,
          ["cal", "cal", "cal", "cal", "list", "play", "rate", "delete"], ORANGE, .21)
    box(ax, .10, .015, .84, .075, "zC: 100% intent-oriented     zS: 100% language-oriented",
        "Cross-language stability: Raw H 0.270  →  zC 0.541", face="#F6FAF7", edge=GREEN, title_color=GREEN, size=8.3, pad=.005)


def main():
    with TRACE_DATA.open(encoding="utf-8") as handle:
        traces = json.load(handle)
    assert len(traces["z_C_334"]) == len(traces["z_S_491"]) == 8
    sns.set_theme(style="white", font="DejaVu Sans")
    plt.rcParams.update({"svg.fonttype": "path", "font.size": 10})
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16.6, 6.0), gridspec_kw={"width_ratios": [1, 1, 1.08]})
    for axis in axes:
        axis.set_axis_off()
    panel_a(axes[0])
    panel_b(axes[1])
    panel_c(axes[2], traces)
    fig.subplots_adjust(left=.025, right=.99, bottom=.06, top=.96, wspace=.10)
    fig.savefig(OUT / "figure2_qualitative_specialization.svg", format="svg")
    plt.close(fig)
    assert (OUT / "figure2_qualitative_specialization.svg").exists()


if __name__ == "__main__":
    main()

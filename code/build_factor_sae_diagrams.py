"""Build the current factor-SAE paper diagrams with Diagram Design defaults."""

import csv
import html
import json
import re
from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "Figures"
PAPER, WHITE, INK = "#f5f5f5", "#ffffff", "#2d3142"
MUTED, SOFT, RULE = "#4f5d75", "#7a8399", "#d8d9de"
ACCENT, ACCENT_TINT, LINK, LINK_TINT = "#eb6c36", "#fcece5", "#2e5aa8", "#e8edf7"
FONT_LINK = "https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500;600&display=swap"


def esc(value):
    return html.escape(str(value), quote=True)


def text(x, y, value, size=12, fill=INK, anchor="start", weight=400, family="Geist", extra=""):
    return (f'<text x="{x}" y="{y}" fill="{fill}" font-size="{size}" font-weight="{weight}" '
            f'font-family="\'{family}\', sans-serif" text-anchor="{anchor}" {extra}>{esc(value)}</text>')


def multiline(x, y, lines, size=12, fill=INK, anchor="middle", weight=400, gap=16, family="Geist"):
    spans = "".join(f'<tspan x="{x}" dy="{0 if i == 0 else gap}">{esc(line)}</tspan>' for i, line in enumerate(lines))
    return (f'<text x="{x}" y="{y}" fill="{fill}" font-size="{size}" font-weight="{weight}" '
            f'font-family="\'{family}\', sans-serif" text-anchor="{anchor}">{spans}</text>')


def box(x, y, w, h, fill=WHITE, stroke=RULE, rx=8, width=1):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'


def arrow(path, color=MUTED, marker="factor-arrow", dash=""):
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2" marker-end="url(#{marker})"{dashed}/>'


def header(title, eyebrow, subtitle=""):
    items = [text(40, 28, eyebrow.upper(), 8, MUTED, weight=600, family="Geist Mono", extra='letter-spacing="1.2"'),
             text(40, 64, title, 28, INK, weight=400, family="Instrument Serif")]
    if subtitle:
        items.append(text(40, 88, subtitle, 12, MUTED))
    return "".join(items)


def svg_document(slug, title_value, description, width, height, body):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="{slug}-title {slug}-desc" viewBox="0 0 {width} {height}">
<title id="{slug}-title">{esc(title_value)}</title>
<desc id="{slug}-desc">{esc(description)}</desc>
<defs>
  <style>@import url('{FONT_LINK.replace('&', '&amp;')}');</style>
  <marker id="factor-arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0,8 3,0 6" fill="{MUTED}"/></marker>
  <marker id="factor-arrow-link" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0,8 3,0 6" fill="{LINK}"/></marker>
  <marker id="factor-arrow-accent" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0,8 3,0 6" fill="{ACCENT}"/></marker>
</defs>
<rect width="{width}" height="{height}" fill="{PAPER}"/>
{body}
</svg>'''


def write_diagram(slug, title_value, description, width, height, body):
    OUT.mkdir(parents=True, exist_ok=True)
    svg = svg_document(slug, title_value, description, width, height, body)
    html_path = OUT / f"{slug}.html"
    svg_path = OUT / f"{slug}.svg"
    html_path.write_text(f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>{esc(title_value)}</title><link href="{FONT_LINK}" rel="stylesheet"><style>html,body{{margin:0;background:{PAPER}}}svg{{display:block;width:100%;height:auto}}</style></head><body>{svg}</body></html>''', encoding="utf-8")
    svg_path.write_text('<?xml version="1.0" encoding="UTF-8"?>\n' + svg, encoding="utf-8")
    return html_path, svg_path, width, height


def figure1():
    parts = [header("A factor-contrastive sparse dictionary", "Method", "Controlled relations organize one overcomplete BatchTopK code.")]
    # Connectors first.
    parts += [
        arrow("M200 252 H240"),
        arrow("M440 236 H468 Q476 236 476 228 V188 Q476 180 484 180 H512", LINK, "factor-arrow-link"),
        arrow("M440 284 H468 Q476 284 476 292 V356 Q476 364 484 364 H512", ACCENT, "factor-arrow-accent"),
        arrow("M712 180 H744 Q752 180 752 188 V228 Q752 236 760 236 H792", LINK, "factor-arrow-link"),
        arrow("M712 364 H744 Q752 364 752 356 V292 Q752 284 760 284 H792", ACCENT, "factor-arrow-accent"),
        arrow("M952 260 H992"),
        arrow("M1152 260 H1184"),
    ]
    # Nodes.
    parts += [
        box(40, 204, 160, 112, WHITE, MUTED),
        text(120, 232, "FROZEN INPUT", 8, MUTED, "middle", 600, "Geist Mono", 'letter-spacing="1"'),
        text(120, 264, "Gemma H", 16, INK, "middle", 600),
        text(120, 288, "2,304 dimensions", 12, MUTED, "middle", family="Geist Mono"),
        box(240, 204, 200, 112, WHITE, INK),
        text(340, 232, "SHARED ENCODER", 8, MUTED, "middle", 600, "Geist Mono", 'letter-spacing="1"'),
        text(340, 264, "Linear + ReLU", 16, INK, "middle", 600),
        text(340, 288, "2,304 → 9,216", 12, MUTED, "middle", family="Geist Mono"),
        box(512, 120, 200, 120, LINK_TINT, LINK, width=2),
        text(612, 148, "INTENT ROUTE", 8, LINK, "middle", 600, "Geist Mono", 'letter-spacing="1"'),
        text(612, 180, "zC · 2,765 features", 16, INK, "middle", 600),
        text(612, 204, "BatchTopK budget B×13", 12, LINK, "middle", family="Geist Mono"),
        text(612, 224, "same intent / different locale", 8, MUTED, "middle"),
        box(512, 304, 200, 120, ACCENT_TINT, ACCENT, width=2),
        text(612, 332, "LOCALE ROUTE", 8, ACCENT, "middle", 600, "Geist Mono", 'letter-spacing="1"'),
        text(612, 364, "zS · 6,451 features", 16, INK, "middle", 600),
        text(612, 388, "BatchTopK budget B×51", 12, ACCENT, "middle", family="Geist Mono"),
        text(612, 408, "same locale / different intent", 8, MUTED, "middle"),
        box(792, 204, 160, 112, WHITE, INK),
        text(872, 232, "SPARSE CODE", 8, MUTED, "middle", 600, "Geist Mono", 'letter-spacing="1"'),
        text(872, 264, "[zC ; zS]", 16, INK, "middle", 600),
        text(872, 288, "mean L0 = 64", 12, MUTED, "middle", family="Geist Mono"),
        box(992, 204, 160, 112, WHITE, MUTED),
        text(1072, 232, "DECODER", 8, MUTED, "middle", 600, "Geist Mono", 'letter-spacing="1"'),
        text(1072, 264, "Reconstruct H", 16, INK, "middle", 600),
        text(1072, 288, "ordinary SAE loss", 12, MUTED, "middle"),
        text(1168, 232, "Ĥ", 20, INK, weight=600),
        box(280, 468, 640, 48, WHITE, RULE),
        text(304, 496, "OBJECTIVE", 8, MUTED, weight=600, family="Geist Mono", extra='letter-spacing="1"'),
        text(408, 496, "reconstruction + λ · reciprocal controlled contrast", 12, INK),
        text(1136, 496, "4× dictionary", 8, MUTED, "end", 600, "Geist Mono"),
    ]
    return write_diagram("factor_sae_figure1_method", "Factor-contrastive sparse autoencoder",
                         "Frozen activations enter a 9,216-feature dictionary split into intent and locale routes with separate BatchTopK activity budgets and reciprocal controlled losses.", 1200, 544, "".join(parts))


def read_rows(name):
    with (ROOT / "Report" / name).open(encoding="utf-8", newline="") as handle:
        return {row["method"]: row for row in csv.DictReader(handle)}


def figure2():
    massive = read_rows("factor_sae_step4_definitive_test_summary.csv")
    mtop = read_rows("factor_sae_step6_mtop_test_summary.csv")
    pythia = read_rows("factor_sae_pythia160m_transfer_summary.csv")
    width8 = read_rows("factor_sae_step5_width8_test_summary.csv")
    panels = [
        ("MASSIVE · GEMMA", "Intent concept AUC", float(massive["Blockwise SAE control"]["intent_concept_auc_mean"]), float(massive["Reciprocal factor SAE"]["intent_concept_auc_mean"]), 1.0),
        ("MASSIVE · GEMMA", "Feature stability", float(massive["Blockwise SAE control"]["cross_locale_stability_mean"]), float(massive["Reciprocal factor SAE"]["cross_locale_stability_mean"]), .25),
        ("MTOP · GEMMA", "Intent concept AUC", float(mtop["Blockwise SAE control"]["intent_concept_auc_mean"]), float(mtop["Reciprocal factor SAE"]["intent_concept_auc_mean"]), 1.0),
        ("MASSIVE · PYTHIA", "Feature stability", float(pythia["Blockwise SAE control"]["cross_locale_stability_mean"]), float(pythia["Reciprocal factor SAE"]["cross_locale_stability_mean"]), .025),
    ]
    parts = [header("Controlled relations improve sparse organization", "Evidence", "Exact blockwise reconstruction control versus the reciprocal factor SAE.")]
    for index, (eyebrow, metric, control, ours, maximum) in enumerate(panels):
        x, y, w, h = 40 + index * 288, 120, 264, 328
        parts += [box(x, y, w, h, WHITE, RULE), text(x + 20, y + 28, eyebrow, 8, MUTED, weight=600, family="Geist Mono", extra='letter-spacing="1"'),
                  text(x + 20, y + 60, metric, 16, INK, weight=600),
                  text(x + 20, y + 92, f"scale 0–{maximum:g}", 8, SOFT, family="Geist Mono")]
        for row, (label, value, color) in enumerate((("Block control", control, MUTED), ("Ours", ours, LINK))):
            by = y + 136 + row * 88
            bar_w = 208 * value / maximum
            parts += [text(x + 20, by, label, 12, INK, weight=600),
                      f'<rect x="{x + 20}" y="{by + 16}" width="208" height="20" rx="4" fill="{RULE}"/>',
                      f'<rect x="{x + 20}" y="{by + 16}" width="{bar_w:.1f}" height="20" rx="4" fill="{color}"/>',
                      text(x + 228, by + 31, f"{value:.4f}", 12, INK, "end", 600, "Geist Mono")]
        parts.append(text(x + 20, y + 304, f"Δ = {ours - control:+.4f}", 12, LINK, weight=600, family="Geist Mono"))
    ours8, control8 = width8["Reciprocal factor SAE"], width8["Blockwise SAE control"]
    parts += [box(40, 472, 1128, 56, ACCENT_TINT, ACCENT),
              text(64, 504, "8× ROBUSTNESS", 8, ACCENT, weight=600, family="Geist Mono", extra='letter-spacing="1"'),
              text(216, 504, f"AUC {float(ours8['intent_concept_auc_mean']):.3f} vs {float(control8['intent_concept_auc_mean']):.3f}  ·  stability {float(ours8['cross_locale_stability_mean']):.3f} vs {float(control8['cross_locale_stability_mean']):.3f}  ·  ranking unchanged", 12, INK)]
    return write_diagram("factor_sae_figure2_evidence", "Sparse-feature evidence across settings",
                         "Four paired comparisons show the reciprocal factor SAE against the exact blockwise reconstruction control on MASSIVE, MTOP, and Pythia, followed by the eightfold-width robustness result.", 1200, 552, "".join(parts))


def heat_color(value):
    t = max(0.0, min(1.0, (float(value) + 1.0) / 4.0)) ** .65
    lo, hi = (245, 245, 245), (45, 49, 66)
    return "#" + "".join(f"{round(a + (b - a) * t):02x}" for a, b in zip(lo, hi))


def figure3():
    payload = json.loads((ROOT / "paper" / "figure_data" / "figure3_factor_stability.json").read_text(encoding="utf-8"))
    q = payload["qualitative"]
    aggregate = payload["aggregate_three_seed"]
    order = (("BatchTopK SAE", "Global BatchTopK"), ("Reciprocal factor SAE", "Ours"), ("Blockwise SAE control", "Blockwise control"))
    parts = [header("Controlled relations stabilize intent features", "Feature evidence", "The top row uses one validation-selected intent; the bottom row summarizes all active features.")]
    for panel, (method, label) in enumerate(order):
        x, y = 40 + panel * 384, 116
        item = q["methods"][method]
        parts += [text(x, y, label, 16, INK, weight=600),
                  text(x, y + 24, f"feature {item['feature']}  ·  r={item['heldout_stability']:.2f}  ·  AUC={item['heldout_auc']:.2f}", 8, MUTED, family="Geist Mono")]
        matrix = item["standardized_activations"]
        for row in range(2):
            parts.append(text(x, y + 76 + row * 44, q.get("held_out_locales", ["ar", "zh"])[row] if "held_out_locales" in q else ("ar" if row == 0 else "zh"), 8, MUTED, "end", family="Geist Mono"))
            for col in range(12):
                cx, cy = x + 16 + col * 28, y + 52 + row * 44
                parts.append(f'<rect x="{cx}" y="{cy}" width="24" height="32" rx="2" fill="{heat_color(matrix[row][col])}"/>')
        boundary = x + 16 + 6 * 28 - 2
        parts.append(f'<line x1="{boundary}" y1="{y + 48}" x2="{boundary}" y2="{y + 132}" stroke="{ACCENT}" stroke-width="2"/>')
        for col in range(12):
            parts.append(text(x + 28 + col * 28, y + 152, ("T" if col < 6 else "O") + str(col + 1 if col < 6 else col - 5), 8, MUTED, "middle", family="Geist Mono"))
    # Aggregate bottom row.
    titles = ("Gemma · mean stability", "Pythia · mean stability", "Intent − locale fraction")
    for panel, title_value in enumerate(titles):
        x, y = 40 + panel * 384, 360
        parts += [box(x, y, 352, 248, WHITE, RULE), text(x + 20, y + 32, title_value, 16, INK, weight=600)]
        if panel < 2:
            backbone = "Gemma 2 2B" if panel == 0 else "Pythia-160M"
            maximum = .25 if panel == 0 else .025
            values = [("Global", aggregate[backbone]["BatchTopK SAE"]["stability_mean"], SOFT),
                      ("Block", aggregate[backbone]["Blockwise SAE control"]["stability_mean"], MUTED),
                      ("Ours", aggregate[backbone]["Reciprocal factor SAE"]["stability_mean"], LINK)]
            for row, (label, value, color) in enumerate(values):
                by = y + 72 + row * 52
                parts += [text(x + 20, by, label, 12, INK, weight=600),
                          f'<rect x="{x + 84}" y="{by - 16}" width="220" height="20" rx="4" fill="{RULE}"/>',
                          f'<rect x="{x + 84}" y="{by - 16}" width="{220 * value / maximum:.1f}" height="20" rx="4" fill="{color}"/>',
                          text(x + 324, by, f"{value:.3f}", 12, INK, "end", 600, "Geist Mono")]
        else:
            for row, backbone in enumerate(("Gemma 2 2B", "Pythia-160M")):
                by = y + 92 + row * 84
                gaps = [("Global", aggregate[backbone]["BatchTopK SAE"]["orientation_gap_mean"], SOFT),
                        ("Block", aggregate[backbone]["Blockwise SAE control"]["orientation_gap_mean"], MUTED),
                        ("Ours", aggregate[backbone]["Reciprocal factor SAE"]["orientation_gap_mean"], LINK)]
                parts.append(text(x + 20, by - 24, "Gemma" if row == 0 else "Pythia", 12, INK, weight=600))
                for col, (label, value, color) in enumerate(gaps):
                    parts += [text(x + 20 + col * 104, by, label, 8, MUTED, family="Geist Mono"),
                              text(x + 20 + col * 104, by + 24, f"{value:+.3f}", 12, color, weight=600, family="Geist Mono")]
    return write_diagram("factor_sae_figure3_stability", "Feature stability across held-out locales",
                         "Top panels compare validation-selected features in global BatchTopK, the reciprocal factor SAE, and the blockwise control; bottom panels report aggregate stability and feature orientation for Gemma and Pythia.", 1200, 640, "".join(parts))


def figure4():
    payload = json.loads((ROOT / "paper" / "figure_data" / "figure4_feature_examples.json").read_text(encoding="utf-8"))
    parts = [header("Stable sparse features recover recognizable intents", "Interpretability", "Three validation-selected features evaluated on held-out Arabic and Chinese.")]
    for index, feature in enumerate(payload["features"]):
        x, y, w, h = 40 + index * 384, 120, 352, 392
        focal = index == 1
        parts += [box(x, y, w, h, ACCENT_TINT if focal else WHITE, ACCENT if focal else RULE, width=2 if focal else 1),
                  text(x + 20, y + 28, feature["intent_name"].replace("_", " ").upper(), 8, ACCENT if focal else MUTED, weight=600, family="Geist Mono", extra='letter-spacing="1"'),
                  text(x + 20, y + 64, f"feature {feature['feature_id']}", 20, INK, weight=600),
                  text(x + 20, y + 92, f"purity@20  {feature['top20_intent_purity']:.2f}", 12, LINK, weight=600, family="Geist Mono"),
                  text(x + 184, y + 92, f"stability  {feature['cross_locale_stability']:.2f}", 12, MUTED, weight=600, family="Geist Mono"),
                  f'<line x1="{x + 20}" y1="{y + 112}" x2="{x + w - 20}" y2="{y + 112}" stroke="{RULE}"/>']
        for row, example in enumerate(feature["examples"]):
            ey = y + 152 + row * 72
            sentence = example["english"]
            if len(sentence) > 45:
                sentence = sentence[:42].rstrip() + "…"
            parts += [text(x + 20, ey, example["locale"], 8, LINK if example["locale"].startswith("zh") else ACCENT, weight=600, family="Geist Mono"),
                      text(x + 20, ey + 24, f'“{sentence}”', 12, INK),
                      text(x + w - 20, ey, f"a={example['activation']:.1f}", 8, MUTED, "end", family="Geist Mono")]
    summary = next(row for row in payload["protocol"]["summary"] if row["method"] == "Reciprocal factor SAE")
    parts += [box(40, 536, 1128, 56, LINK_TINT, LINK),
              text(64, 568, "3-SEED SUMMARY", 8, LINK, weight=600, family="Geist Mono", extra='letter-spacing="1"'),
              text(224, 568, f"purity {summary['mean_top20_intent_purity_mean']:.3f}  ·  reliable coverage {summary['reliable_intent_coverage_mean']:.3f}  ·  selected-feature stability {summary['mean_selected_feature_stability_mean']:.3f}", 12, INK)]
    return write_diagram("factor_sae_figure4_examples", "Held-out intent-feature examples",
                         "Three sparse features for takeaway ordering, coffee control, and currency questions show high intent purity and cross-locale stability with representative held-out examples.", 1200, 616, "".join(parts))


def export(diagrams):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        for html_path, _, width, height in diagrams:
            page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=2)
            page.goto(html_path.resolve().as_uri(), wait_until="load")
            page.locator("svg").screenshot(path=str(html_path.with_suffix(".png")), omit_background=True)
            page.pdf(path=str(html_path.with_suffix(".pdf")), width=f"{width}px", height=f"{height}px",
                     print_background=True, margin={"top": "0", "right": "0", "bottom": "0", "left": "0"})
            page.close()
        browser.close()


def main():
    diagrams = [figure1(), figure2(), figure3(), figure4()]
    export(diagrams)
    for html_path, svg_path, _, _ in diagrams:
        source = html_path.read_text(encoding="utf-8")
        assert re.search(r'<svg[^>]+role="img"[^>]+aria-labelledby=', source)
        assert source.index("<title") < source.index("<defs")
        assert svg_path.exists() and html_path.with_suffix(".pdf").exists() and html_path.with_suffix(".png").exists()
        print(html_path)


if __name__ == "__main__":
    main()

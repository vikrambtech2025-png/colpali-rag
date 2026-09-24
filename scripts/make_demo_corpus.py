"""Generate a small demo corpus: PDFs with text + figures + tables (visual retrieval demos).

Run: uv run python -m scripts.make_demo_corpus
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "corpus"


def _plt_close(fig) -> None:
    plt.close(fig)


def make_revenue_chart() -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.4))
    quarters = ["Q1", "Q2", "Q3", "Q4"]
    products = ["Cloud", "Devices", "Services"]
    data = {"Cloud": [12, 15, 18, 22], "Devices": [9, 8, 7, 6], "Services": [4, 5, 7, 9]}
    bottom = np.zeros(4)
    colors = ["#3ba55d", "#4a90d9", "#e8a33d"]
    for i, name in enumerate(products):
        ax.bar(quarters, data[name], bottom=bottom, label=name, color=colors[i], width=0.58)
        bottom += np.array(data[name])
    ax.set_title("Quarterly Revenue by Product Line (USD millions)")
    ax.set_ylabel("USD m")
    ax.legend()
    img = OUT / "fig-revenue.png"
    fig.savefig(img, dpi=130, bbox_inches="tight")
    _plt_close(fig)
    return img


def make_attention_figure() -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.4))
    steps = np.arange(0, 10)
    ax.plot(steps, np.sin(steps * 0.8), "o-", color="#3ba55d", label="self-attention score")
    ax.plot(steps, np.abs(np.cos(steps * 0.6)) * 0.4, "s--", color="#4a90d9", label="cross-attention gate")
    ax.set_title("Attention weight behavior over decoding steps")
    ax.set_xlabel("decoding step")
    ax.set_ylabel("normalized weight")
    ax.legend()
    img = OUT / "fig-attention.png"
    fig.savefig(img, dpi=130, bbox_inches="tight")
    _plt_close(fig)
    return img


def make_battery_scatter() -> Path:
    rng = np.random.default_rng(7)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = rng.uniform(10, 90, 60)  # cost per kWh
    y = rng.uniform(180, 420, 60)  # energy density Wh/kg
    sizes = rng.uniform(40, 220, 60)
    ax.scatter(x, y, s=sizes, alpha=0.55, color="#3ba55d", edgecolor="none")
    ax.axvline(50, color="#e8a33d", ls="--", lw=1, label="cost target $50/kWh")
    ax.set_title("Battery chemistries: energy density vs cost")
    ax.set_xlabel("cost (USD per kWh)")
    ax.set_ylabel("energy density (Wh/kg)")
    ax.legend()
    img = OUT / "fig-battery.png"
    fig.savefig(img, dpi=130, bbox_inches="tight")
    _plt_close(fig)
    return img


def build_report_pdf(figs: dict[str, Path]) -> Path:
    import fitz

    pdf_path = OUT / "Q3-2025-Market-Intelligence-Report.pdf"
    docs = fitz.open()
    # page 1: cover / exec summary (text only)
    p1 = docs.new_page(width=595, height=842)
    p1.insert_textbox(fitz.Rect(60, 90, 535, 300), "Q3 2025 Market Intelligence Report\n\nExecutive Summary\n\nRevenue grew to $37M in Q3 2025, led by the Cloud product line (22M USD). Devices declined slightly while Services accelerated after the model-licensing deal announced in August.", fontsize=13)
    # page 2: revenue chart figure on left, text on right
    p2 = docs.new_page(width=595, height=842)
    p2.insert_image(fitz.Rect(40, 90, 380, 330), filename=str(figs["revenue"]))
    p2.insert_textbox(fitz.Rect(40, 380, 555, 620), "The Cloud line passed Devices in Q2 and widened the gap through Q3, ending the quarter at $22M. Devices declined from $9M to $6M as the consumer refresh slipped to Q4. Services posted the strongest growth rate, +29% quarter over quarter, reaching $9M on the strength of managed-model revenue.", fontsize=11)
    # page 3: battery chemistry scatter + table
    p3 = docs.new_page(width=595, height=842)
    p3.insert_image(fitz.Rect(40, 90, 380, 330), filename=str(figs["battery"]))
    table_rows = [
        ("Chemistry", "Cost $/kWh", "Density Wh/kg"),
        ("LFP", "45", "210"),
        ("NMC 811", "62", "285"),
        ("Solid-state (demo)", "240", "420"),
    ]
    y = 420
    for row in table_rows:
        p3.insert_text(fitz.Point(60, y), "  |  ".join(row), fontsize=11)
        y += 26
    p3.insert_text(fitz.Point(60, 700), "Source: internal teardown lab, Sep 2025. Solid-state remains a demo-line chemistry.", fontsize=9)
    # page 4: competitive landscape text
    p4 = docs.new_page(width=595, height=842)
    p4.insert_textbox(fitz.Rect(60, 90, 535, 700), "Competitive landscape\n\nThree clusters dominate: hyperscalers bundling models into platforms, open-weight labs monetizing licenses, and managed-service vendors. Our strategic position sits at the managed-service boundary, where the licensing deal deepens margin. Risk: devices supply chain concentration remains in one assembly partner.", fontsize=11)
    docs.save(str(pdf_path))
    docs.close()
    return pdf_path


def build_notes_pdf(fig: Path) -> Path:
    import fitz

    pdf_path = OUT / "Transformer-Architecture-Field-Notes.pdf"
    docs = fitz.open()
    p1 = docs.new_page(width=595, height=842)
    p1.insert_textbox(fitz.Rect(60, 90, 535, 500), "Transformer Architecture Field Notes\n\nAttention mechanisms can be grouped by where they gate information. Self-attention computes a weighted mixture over the input sequence; cross-attention instead mixes over an encoder memory. The figure to the right plots observed gate behavior during decoding.", fontsize=12)
    p2 = docs.new_page(width=595, height=842)
    p2.insert_image(fitz.Rect(40, 90, 380, 330), filename=str(fig))
    p2.insert_textbox(fitz.Rect(40, 380, 555, 800), "Observations: the self-attention score oscillates with the token position, while the cross-attention gate stays low except at structure boundaries. In practice, clamping the cross-attention gate below 0.35 avoids attention collapse in long decoding runs. KV-cache reuse is safe above step 4 in this configuration.", fontsize=11)
    docs.save(str(pdf_path))
    docs.close()
    return pdf_path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    figs = {
        "revenue": make_revenue_chart(),
        "attention": make_attention_figure(),
        "battery": make_battery_scatter(),
    }
    r1 = build_report_pdf(figs)
    r2 = build_notes_pdf(figs["attention"])
    print(f"Demo corpus written to {OUT}")
    print(f"  1. {r1.name}")
    print(f"  2. {r2.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
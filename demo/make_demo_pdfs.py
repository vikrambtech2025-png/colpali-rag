"""Generate the college-showcase demo corpus: polished PDFs with real charts,
tables and figures (the kind of docs that show off vision-first retrieval).

Usage:
    python demo/make_demo_pdfs.py            # writes demo/pdfs/<name>.pdf

Notes:
- Every page carries REAL embedded text (so the dense/sparse legs have signal)
  AND a full-page rendered figure where relevant (so ColPali wins chart/figure
  questions — the demo moment).
- Requires: matplotlib, pymupdf. Pure CPU; ~2s to run.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pymupdf as fitz  # noqa: E402

OUT = Path(__file__).resolve().parent / "pdfs"

# ---- shared styling -------------------------------------------------------
NAVY = (0.09, 0.25, 0.42)
TEAL = (0.14, 0.55, 0.55)
AMBER = (0.93, 0.75, 0.25)
RED = (0.78, 0.28, 0.30)
DARK = (0.16, 0.20, 0.26)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.grid": True,
        "grid.color": "#dde3ea",
        "grid.linewidth": 0.7,
    }
)


def _fig_to_png(fig, name: str) -> Path:
    path = OUT / f"_fig-{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _new_page(doc: fitz.Document, title: str, subtitle: str = "") -> fitz.Page:
    page = doc.new_page(width=612, height=792)  # US Letter
    page.draw_rect(fitz.Rect(0, 0, 612, 54), color=None, fill=NAVY)
    page.insert_text((34, 30), title, fontsize=17, fontname="helv", color=(1, 1, 1))
    if subtitle:
        page.insert_text((34, 48), subtitle, fontsize=9, fontname="helv", color=(0.80, 0.88, 0.96))
    return page


def _paragraph(page: fitz.Page, text: str, y: float, size: float = 10.5) -> float:
    """Justified paragraph; returns the next free y."""
    rect = fitz.Rect(40, y, 572, 760)
    page.insert_textbox(rect, text, fontsize=size, fontname="helv", color=DARK, align=fitz.TEXT_ALIGN_JUSTIFY)
    # measure: rough height = chars / ~105 per line * line height
    lines = max(1, int(len(text) / 108))
    return y + lines * (size + 4.6)


def _table(page: fitz.Page, headers: list[str], rows: list[list[str]], y: float, widths: list[float], col_colors: list[tuple] | None = None, header_fill=NAVY) -> float:
    """Plain-text table (extractable for the text leg) with a colored header band."""
    x = 40.0
    col_x: list[float] = []
    for w in widths:
        col_x.append(x)
        x += w
    page.draw_rect(fitz.Rect(40, y - 14, 572, y + 2), color=None, fill=header_fill)
    for cx, h in zip(col_x, headers):
        page.insert_text((cx + 4, y), h, fontsize=9.5, fontname="helv", color=(1, 1, 1))
    yy = y + 14
    for idx, row in enumerate(rows):
        if col_colors and idx % 2 == 0:
            page.draw_rect(fitz.Rect(40, yy - 13, 572, yy + 1), color=None, fill=(0.93, 0.95, 0.98))
        for cx, cell in zip(col_x, row):
            page.insert_text((cx + 4, yy), cell, fontsize=9.5, fontname="cour", color=DARK)
        yy += 15
    return yy + 8


def _chart_page(doc: fitz.Document, title: str, subtitle: str, fig, caption: str, intro: str) -> None:
    page = _new_page(doc, title, subtitle)
    y = 74
    if intro:
        y = _paragraph(page, intro, y) + 10
    png = _fig_to_png(fig, title.replace(" ", "_").replace("/", "_"))
    rect = fitz.Rect(42, y, 570, y + 330)
    page.insert_image(rect, filename=str(png), keep_proportion=True)
    page.insert_text((44, 760), caption, fontsize=8.5, fontname="helv", color=(0.35, 0.42, 0.52))
    png.unlink(missing_ok=True)


# ---- Document 1: NanoSat constellation spec --------------------------------
def make_nanosat() -> list[Path]:
    doc = fitz.open()

    # p1 cover + system overview with a block diagram
    page = _new_page(doc, "NanoSat Constellation Design Specification", "Revision 4.2 - Systems Engineering Office")
    y = _paragraph(
        page,
        "Project KESTREL is a 24-satellite autonomous nano-satellite constellation for low-latency Earth observation. "
        "Each satellite carries a 3-band multispectral imager, an S-band downlink, and an on-board stereo vision processor "
        "for cloud screening. The constellation is designed to provide a revisit interval below 30 minutes across mid "
        "latitudes, which is the key requirement inherited from the disaster-response program office.",
        74,
    )
    boxes = [
        ("SENSOR BUS", ["3-band imager", "8m GSD", "200 km swath"], 40, y + 6, 150, 90, TEAL),
        ("EDGE AI", ["cloud screening", "on-board stereo vision", "RISC-V 8-core"], 210, y + 6, 150, 90, (0.13, 0.44, 0.64)),
        ("DOWNLINK", ["S-band 2.4 Gbps", "daily contact budget", "3 ground stations"], 380, y + 6, 150, 90, AMBER),
    ]
    for name, lines, bx, by, bw, bh, col in boxes:
        page.draw_rect(fitz.Rect(bx, by, bx + bw, by + bh), color=None, fill=col)
        page.insert_text((bx + 10, by + 20), name, fontsize=10, fontname="helv", color=(1, 1, 1))
        for i, ln in enumerate(lines):
            page.insert_text((bx + 10, by + 40 + i * 14), ln, fontsize=8.5, fontname="helv", color=(1, 1, 1))
    page.insert_text((150, 304), "24 satellites | 540 km dusk-dawn SSO | 8 m GSD | < 30 min revisit", fontsize=10, fontname="helv", color=DARK)
    y2 = _paragraph(
        page,
        "The satellite bus is derived from the flight-proven 3U CubeSat platform with an extendable solar array. "
        "Each unit carries redundant reaction wheels and a cold-gas propulsion module used for de-orbit compliance. "
        "The system software runs a real-time Linux core with a watchdog partition that restarts the imaging pipeline "
        "within 4 seconds of any detected fault.",
        340,
    )
    _table(
        page,
        ["Item", "Value", "Note"],
        [
            ["Orbit", "540 km SSO, 97.6 deg", "dusk-dawn, LTAN 18:00"],
            ["Revisit", "< 30 min (mid-lat)", "24 sats, 4 planes x 6"],
            ["Imager GSD", "8 m (RGB+NIR)", "3-band + pan sharpen"],
            ["Downlink", "S-band 2.4 Gbps", "3 x 7.3m ground stations"],
            ["Edge AI", "8-core RISC-V", "on-board cloud screen"],
        ],
        y2 + 40,
        [110, 170, 200],
    )

    # p2 power budget (table-heavy page)
    page = _new_page(doc, "Power Budget & Energy Balance", "Worst-case eclipse season, 3U bus")
    y = _paragraph(
        page,
        "The energy balance is dominated by the imager duty cycle and the S-band transmission window. During the worst "
        "eclipse season the satellite survives on battery alone for 34 minutes of each orbit. The extended solar array "
        "provides 312 Wh per orbit day at end of life, which closes the budget with a 23 percent margin across all "
        "operating modes.",
        74,
    )
    yy = _table(
        page,
        ["Component", "Power (W)", "Duty", "Avg (W)"],
        [
            ["Imager + shutter", "14.2", "15%", "2.1"],
            ["Edge AI (RISC-V)", "6.8", "60%", "4.1"],
            ["S-band TX", "38.0", "8%", "3.0"],
            ["OBC + RTOS", "2.4", "100%", "2.4"],
            ["Attitude control", "1.9", "40%", "0.8"],
            ["Thermal + heaters", "3.1", "12%", "0.4"],
            ["TOTAL (orbit avg)", "-", "-", "12.8"],
        ],
        y + 10,
        [190, 110, 80, 100],
    )
    page.insert_text((40, yy + 4), "Battery: 6S 18650 pack, 156 Wh usable, 90% depth of discharge.", fontsize=10, fontname="helv", color=DARK)
    _paragraph(page, "The 23% margin is held against a 10% solar-array degradation assumption at end of life, plus a "
        "worst-case 5% pointing loss during momentum dumps. The sizing is verified with a Monte-Carlo orbit propagator "
        "over 1,000 sampled launch epochs.", yy + 36)

    # p3 link budget chart
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    alt = np.arange(400, 760, 20)
    link_margin = 14.0 - 0.028 * (alt - 400) + 1.2 * np.sin(alt / 90)
    ax.plot(alt, link_margin, color="#20639B", lw=2.6, marker="o", ms=3)
    ax.axhline(3.0, color="#E07A5F", ls="--", lw=1.6, label="minimum margin (3 dB)")
    ax.set_xlabel("Orbit altitude (km)")
    ax.set_ylabel("S-band link margin (dB)")
    ax.set_title("Downlink margin vs altitude at 25 deg elevation")
    ax.legend(frameon=False)
    _chart_page(
        doc, "RF Link Budget", "S-band downlink, 2.4 Gbps, 25 deg mask",
        fig,
        "Margin stays above 11 dB across the altitude band; the 540 km design point sits at 14.9 dB.",
        "The link budget closes at all altitudes between 400 and 750 km with at least 11.3 dB of margin against the "
        "3 dB minimum, leaving headroom for rain fade and pointing error during passes over the pole.",
    )

    # p4 risk matrix (colored table)
    page = _new_page(doc, "Risk Register", "Likelihood x impact matrix - launch + LEO ops")
    y = _paragraph(
        page,
        "The risk posture over the first 18 months of operations is dominated by launch-slot slip and propulsion "
        "anomalies. Two items exceed the red threshold and are actively mitigated with hardware redundancy and a "
        "de-orbiting waiver review.",
        74,
    )
    risks = [
        ("Launch slot slip", "L2, I3", (0.93, 0.83, 0.30)),
        ("Propulsion anomaly", "L2, I4", (0.91, 0.45, 0.42)),
        ("Imager calibration drift", "L3, I2", (0.93, 0.83, 0.30)),
        ("S-band interference", "L3, I3", (0.93, 0.83, 0.30)),
        ("Sun-synchronous drift", "L4, I2", (0.77, 0.86, 0.62)),
        ("Ground station outage", "L4, I3", (0.93, 0.83, 0.30)),
        ("Debris conjunction", "L2, I4", (0.91, 0.45, 0.42)),
    ]
    yy = y + 6
    page.draw_rect(fitz.Rect(40, yy - 14, 572, yy + 2), color=None, fill=NAVY)
    page.insert_text((44, yy), "Risk item", fontsize=9.5, fontname="helv", color=(1, 1, 1))
    page.insert_text((320, yy), "L x I rating", fontsize=9.5, fontname="helv", color=(1, 1, 1))
    page.insert_text((430, yy), "Status", fontsize=9.5, fontname="helv", color=(1, 1, 1))
    yy += 15
    for name, rating, col in risks:
        page.draw_rect(fitz.Rect(40, yy - 13, 572, yy + 2), color=None, fill=col)
        page.insert_text((44, yy), name, fontsize=9.5, fontname="helv", color=DARK)
        page.insert_text((320, yy), rating, fontsize=9.5, fontname="cour", color=DARK)
        page.insert_text((430, yy), "mitigate" if rating in ("L2, I4",) else "watch", fontsize=9.5, fontname="helv", color=DARK)
        yy += 17
    _paragraph(page, "Mitigations: the propulsion module is dual-string with a cold backup; imager gain is re-calibrated "
        "against a lunar target every 21 days; the de-orbiting waiver review is scheduled 60 days after first orbit.",
        yy + 14)

    out = OUT / "NanoSat-Constellation-Design-Spec.pdf"
    doc.save(str(out))
    doc.close()
    return [out]


# ---- Document 2: SaaS cohort report -----------------------------------------
def make_saas() -> list[Path]:
    doc = fitz.open()

    # p1 retention curves
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    months = np.arange(0, 12)
    cohorts = {
        "Jan 2026 launch": 100 * np.exp(-months / 3.6),
        "Apr 2026 onboarding v2": 100 * np.exp(-months / 6.1),
        "Jul 2026 pricing test": 100 * np.exp(-months / 5.2),
    }
    colors = ["#8E7CC3", "#3BB273", "#E1BC29"]
    for (name, vals), col in zip(cohorts.items(), colors):
        ax.plot(months, vals, marker="o", ms=4, lw=2.4, color=col, label=name)
    ax.set_xlabel("Months since first activation")
    ax.set_ylabel("Retention (%)")
    ax.set_title("Cohort retention - the onboarding v2 lift is visible after month 4")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False)
    _chart_page(
        doc, "Cohort Retention Analysis", "Quarterly retention deep-dive, subscription business",
        fig,
        "The April onboarding refresh lifted month-6 retention from 28% to 49% - the biggest lever identified this year.",
        "This report tracks activation-to-churn behavior for the subscription product. The headline finding is that the "
        "April onboarding refresh increased twelve-month expected retention by 41 percent relative to the January launch "
        "cohort. The effect compounds starting at month four, which matches the activation window of 30 days and the "
        "first renewal cycle.",
    )

    # p2 activation funnel (chart) + notes
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    stages = ["Signed up", "First login", "Activated API", "First query", "Paid plan"]
    vals = [100, 84, 61, 44, 38]
    cols = ["#8E7CC3", "#6C7AE0", "#3BB273", "#2E9E67", "#1F7A50"]
    ax.barh(stages[::-1], vals[::-1], color=cols[::-1])
    for i, v in enumerate(vals[::-1]):
        ax.text(v + 1, i, f"{v}%", va="center", fontsize=11, color=DARK)
    ax.set_xlim(0, 115)
    ax.set_xlabel("Percent of new signups")
    ax.set_title("Activation funnel - biggest drop is the API activation step")
    _chart_page(
        doc, "Activation Funnel", "Where new users drop off, Jan-Jun 2026",
        fig,
        "Only 61% of signups reach a working API call; fixing the activation step is worth an estimated +9% conversion.",
        "The funnel shows the single largest drop between first login and a successful API call. Users who activated "
        "within 48 hours of signup retained at 71 percent after three months, versus 34 percent for late activators. "
        "The onboarding team is trialing an inline activation code flow to compress this step.",
    )

    # p3 cohort table (numeric)
    page = _new_page(doc, "Cohort Retention Table", "Monthly retention by activation cohort")
    y = _paragraph(
        page,
        "The table below gives month-by-month retention for each onboarding cohort. Cohorts activated with the "
        "simplified inline flow (v2) retain materially better from month two onward. The July cohort is still young "
        "and should be read cautiously.",
        74,
    )
    _table(
        page,
        ["Cohort", "M0", "M1", "M3", "M6", "M12"],
        [
            ["Jan (launch v1)", "100%", "72%", "41%", "28%", "19%"],
            ["Apr (onboarding v2)", "100%", "81%", "63%", "49%", "31%"],
            ["Jul (pricing test)", "100%", "78%", "58%", "-", "-"],
            ["Oct (current)", "100%", "80%", "-", "-", "-"],
        ],
        y + 10,
        [160, 70, 70, 70, 70, 70],
    )
    _paragraph(page, "The April v2 cohort is the control-beating case: +21 points of retention at month six. "
        "Qualitatively, churn reasons shifted from 'could not figure out activation' to 'workflow mismatch', which "
        "supports the activation-first hypothesis.", y + 150)

    # p4 what moved the metric
    page = _new_page(doc, "Findings & Recommendations", "Summary for the leadership review")
    y = _paragraph(
        page,
        "Three changes moved retention this quarter. First, the inline activation code replaced the email-and-CLI "
        "flow and lifted activation from 48 to 61 percent of signups. Second, a 90-day usage-nudge sequence reduced "
        "month-one churn by six points. Third, removing the annual-plan paywall from the trial tier increased the "
        "trial-to-paid conversion by nine percent without diluting ARPU.",
        74,
    )
    _table(
        page,
        ["Lever", "Impact", "Confidence"],
        [
            ["Inline activation code", "+13 pt activation", "high"],
            ["90-day nudge sequence", "-6 pt M1 churn", "medium"],
            ["Annual plan in trial", "+9 pt trial->paid", "medium"],
            ["Pricing test (Jul)", "neutral so far", "low"],
        ],
        y + 40,
        [190, 160, 110],
    )
    _paragraph(page, "Recommended next step: instrument the activation step with an in-product checklist and ship the "
        "inline flow to 100% of signups by November, then re-run this cohort analysis in January to confirm the lift "
        "holds outside the launch quarter.", y + 190)

    out = OUT / "SaaS-Cohort-Retention-Report.pdf"
    doc.save(str(out))
    doc.close()
    return [out]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []
    made += make_nanosat()
    made += make_saas()
    for p in made:
        print(f"wrote {p} ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
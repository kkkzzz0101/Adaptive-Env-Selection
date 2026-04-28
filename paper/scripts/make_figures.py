#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
from pathlib import Path

from reportlab.lib import colors
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "paper" / "figures"

BLUE = colors.HexColor("#4C78A8")
ORANGE = colors.HexColor("#F58518")
GREEN = colors.HexColor("#54A24B")
RED = colors.HexColor("#E45756")
DARK_RED = colors.HexColor("#B91C1C")
PURPLE = colors.HexColor("#7E57C2")
GRAY = colors.HexColor("#6B7280")
DARK = colors.HexColor("#374151")
LIGHT_GRAY = colors.HexColor("#F3F4F6")


def ensure_dir() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def convert_png(pdf_path: Path) -> None:
    png_path = pdf_path.with_suffix(".png")
    convert = shutil.which("convert")
    if not convert:
        return
    subprocess.run(
        [convert, "-density", "220", str(pdf_path), "-quality", "95", str(png_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def new_canvas(name: str, width: float, height: float) -> tuple[canvas.Canvas, Path]:
    ensure_dir()
    path = FIG_DIR / f"{name}.pdf"
    c = canvas.Canvas(str(path), pagesize=(width, height))
    c.setTitle(name)
    c.setFillColor(colors.white)
    c.rect(0, 0, width, height, stroke=0, fill=1)
    return c, path


def finish(c: canvas.Canvas, path: Path) -> None:
    c.showPage()
    c.save()
    convert_png(path)
    print(f"[saved] {path}")
    if path.with_suffix(".png").exists():
        print(f"[saved] {path.with_suffix('.png')}")


def text(c: canvas.Canvas, x: float, y: float, s: str, size: int = 9, bold: bool = False, color=DARK, anchor="middle") -> None:
    c.setFillColor(color)
    c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
    if anchor == "middle":
        c.drawCentredString(x, y, s)
    elif anchor == "right":
        c.drawRightString(x, y, s)
    else:
        c.drawString(x, y, s)


def box(c: canvas.Canvas, x: float, y: float, w: float, h: float, label: str, fill=colors.white) -> None:
    c.setStrokeColor(DARK)
    c.setFillColor(fill)
    c.roundRect(x, y, w, h, 8, stroke=1, fill=1)
    lines = label.split("\n")
    start = y + h / 2 + (len(lines) - 1) * 6
    for i, line in enumerate(lines):
        text(c, x + w / 2, start - i * 12, line, size=8)


def arrow(c: canvas.Canvas, x1: float, y1: float, x2: float, y2: float, color=DARK) -> None:
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(1.1)
    c.line(x1, y1, x2, y2)
    angle = math.atan2(y2 - y1, x2 - x1)
    head = 7
    a1 = angle + math.pi * 0.82
    a2 = angle - math.pi * 0.82
    p1 = (x2 + head * math.cos(a1), y2 + head * math.sin(a1))
    p2 = (x2 + head * math.cos(a2), y2 + head * math.sin(a2))
    c.line(x2, y2, p1[0], p1[1])
    c.line(x2, y2, p2[0], p2[1])


def draw_axes(c: canvas.Canvas, x: float, y: float, w: float, h: float, ymax: float, ylabel: str) -> None:
    c.setStrokeColor(DARK)
    c.setLineWidth(0.8)
    c.line(x, y, x, y + h)
    c.line(x, y, x + w, y)
    c.setStrokeColor(colors.HexColor("#D1D5DB"))
    c.setLineWidth(0.4)
    for t in range(0, 6):
        yy = y + h * t / 5
        c.line(x, yy, x + w, yy)
        val = ymax * t / 5
        text(c, x - 6, yy - 3, f"{val:.1f}" if ymax > 1 else f"{val:.2f}", size=7, color=GRAY, anchor="right")
    if ylabel:
        text(c, x, y + h + 10, ylabel, size=8, color=DARK, anchor="start")


def load_step200() -> dict[str, dict[str, float]]:
    path = ROOT / "experiments/results/math_zebra_2data/baseline_vs_norebucket_metrics.csv"
    out: dict[str, dict[str, float]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            if int(row["step"]) == 200:
                out[row["run"]] = {
                    "math_train": float(row["math_train"]),
                    "zebra_train": float(row["zebra_train"]),
                }
    return out


def weighted_pass_rate(path: Path) -> float:
    total = 0.0
    n_total = 0.0
    with path.open() as f:
        for row in csv.DictReader(f):
            n = float(row["n"])
            total += n * float(row["task_pass_rate"])
            n_total += n
    return total / n_total


def fig_method_pipeline() -> None:
    c, path = new_canvas("fig_method_pipeline", 560, 210)
    text(c, 280, 188, "Adaptive Environment Selection feedback loop", size=12, bold=True)

    box(c, 18, 120, 104, 48, "Math + Zebra\ntasks", fill=colors.HexColor("#EFF6FF"))
    box(c, 150, 120, 104, 48, "Initial difficulty\nbuckets", fill=colors.HexColor("#F0FDF4"))
    box(c, 282, 120, 104, 48, "UCB sampler\n+ prob floor", fill=colors.HexColor("#FFF7ED"))
    box(c, 414, 120, 104, 48, "GRPO / verl\ntraining", fill=colors.HexColor("#FDF2F8"))
    box(c, 414, 32, 104, 48, "Observed\n|advantage|", fill=LIGHT_GRAY)
    box(c, 282, 32, 104, 48, "Decayed micro-\nbucket history", fill=LIGHT_GRAY)
    box(c, 150, 32, 104, 48, "Neighbor-only\nre-bucketing", fill=LIGHT_GRAY)

    arrow(c, 122, 144, 150, 144)
    arrow(c, 254, 144, 282, 144)
    arrow(c, 386, 144, 414, 144)
    arrow(c, 466, 120, 466, 80)
    arrow(c, 414, 56, 386, 56)
    arrow(c, 282, 56, 254, 56)
    arrow(c, 202, 80, 202, 120)
    arrow(c, 150, 48, 80, 120, color=GRAY)
    text(c, 88, 66, "bucket", size=7, color=GRAY)
    text(c, 88, 56, "updates", size=7, color=GRAY)
    finish(c, path)


def fig_step200_math_zebra() -> None:
    data = load_step200()
    baseline = [data["baseline_random"]["math_train"], data["baseline_random"]["zebra_train"]]
    scheduler = [data["scheduler_no_rebucket"]["math_train"], data["scheduler_no_rebucket"]["zebra_train"]]
    labels = ["Math", "Zebra"]

    c, path = new_canvas("fig_step200_math_zebra", 360, 235)
    text(c, 180, 214, "Step-200 Math+Zebra validation", size=11, bold=True)
    x0, y0, w, h = 55, 42, 270, 145
    ymax = 0.62
    draw_axes(c, x0, y0, w, h, ymax, "Validation accuracy")

    group_w = w / len(labels)
    bar_w = 30
    for i, label in enumerate(labels):
        cx = x0 + group_w * (i + 0.5)
        vals = [(baseline[i], BLUE, "Random"), (scheduler[i], ORANGE, "AES")]
        for j, (val, col, _) in enumerate(vals):
            bx = cx + (-bar_w - 3 if j == 0 else 3)
            bh = h * val / ymax
            c.setFillColor(col)
            c.rect(bx, y0, bar_w, bh, stroke=0, fill=1)
            text(c, bx + bar_w / 2, y0 + bh + 6, f"{val:.3f}", size=7)
        text(c, cx, y0 - 16, label, size=9)
        text(c, cx, y0 + h * max(baseline[i], scheduler[i]) / ymax + 28, f"+{scheduler[i]-baseline[i]:.3f}", size=8, bold=True, color=GREEN)

    c.setFillColor(BLUE)
    c.rect(218, 198, 9, 9, stroke=0, fill=1)
    text(c, 231, 199, "Random", size=8, anchor="start")
    c.setFillColor(ORANGE)
    c.rect(276, 198, 9, 9, stroke=0, fill=1)
    text(c, 289, 199, "AES", size=8, anchor="start")
    finish(c, path)


def fig_initial_accuracy_profile() -> None:
    difficulties = [1, 2, 3, 4, 5]
    math_acc = [0.70, 0.60, 0.35, 0.35, 0.10]
    zebra_acc = [0.30, 0.20, 0.25, 0.15, None]

    c, path = new_canvas("fig_initial_accuracy_profile", 410, 250)
    text(c, 205, 230, "Initial accuracy test by nominal difficulty", size=11, bold=True)
    x0, y0, w, h = 58, 45, 300, 145
    ymax = 0.75
    draw_axes(c, x0, y0, w, h, ymax, "Initial accuracy")

    def xy(d, val):
        return x0 + (d - 1) / 4 * w, y0 + val / ymax * h

    for vals, col, label in [(math_acc, BLUE, "Math"), (zebra_acc, RED, "Zebra")]:
        pts = []
        for d, val in zip(difficulties, vals):
            if val is None:
                continue
            pts.append(xy(d, val))
        c.setStrokeColor(col)
        c.setFillColor(col)
        c.setLineWidth(1.4)
        for i in range(len(pts) - 1):
            c.line(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])
        for x, y in pts:
            c.circle(x, y, 3.0, stroke=0, fill=1)
        text(c, pts[-1][0] + 10, pts[-1][1] - 2, label, size=8, color=col, anchor="start")

    for d in difficulties:
        xx = x0 + (d - 1) / 4 * w
        text(c, xx, y0 - 15, f"d{d}", size=8)
    text(c, x0 + w / 2, y0 - 31, "Nominal difficulty label", size=8)
    finish(c, path)


def fig_ucb_score_drift() -> None:
    steps = [60, 80, 100, 120, 140, 160, 180, 200]
    scores = {
        "C0": [0.329, 0.250, 0.190, 0.135, 0.072, 0.101, 0.034, 0.081],
        "C1": [0.301, 0.264, 0.210, 0.189, 0.148, 0.141, 0.106, 0.101],
        "C2": [0.259, 0.243, 0.250, 0.204, 0.176, 0.184, 0.197, 0.182],
        "C3": [0.238, 0.252, 0.256, 0.275, 0.275, 0.233, 0.225, 0.295],
        "C4": [0.230, 0.235, 0.220, 0.216, 0.257, 0.255, 0.264, 0.253],
    }
    palette = {
        "C0": BLUE,
        "C1": ORANGE,
        "C2": GREEN,
        "C3": RED,
        "C4": PURPLE,
    }

    c, path = new_canvas("fig_ucb_score_drift", 500, 275)
    text(c, 250, 254, "UCB score drift across training", size=11, bold=True)
    text(c, 250, 239, "Cluster ranking changes over time, so the sampler is active rather than fixed.", size=8, color=GRAY)

    x0, y0, w, h = 58, 50, 360, 145
    ymax = 0.35
    draw_axes(c, x0, y0, w, h, ymax, "UCB score")

    def xy(step, val):
        return x0 + (step - min(steps)) / (max(steps) - min(steps)) * w, y0 + val / ymax * h

    for label, vals in scores.items():
        pts = [xy(step, val) for step, val in zip(steps, vals)]
        c.setStrokeColor(palette[label])
        c.setFillColor(palette[label])
        c.setLineWidth(1.25)
        for i in range(len(pts) - 1):
            c.line(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])
        for x, y in pts:
            c.circle(x, y, 2.3, stroke=0, fill=1)

    for step in steps:
        xx = x0 + (step - min(steps)) / (max(steps) - min(steps)) * w
        text(c, xx, y0 - 14, str(step), size=7)
    text(c, x0 + w / 2, y0 - 30, "Training step", size=8)

    legend_x = 435
    legend_y = 184
    for i, label in enumerate(scores):
        yy = legend_y - i * 17
        c.setStrokeColor(palette[label])
        c.setFillColor(palette[label])
        c.line(legend_x, yy, legend_x + 18, yy)
        c.circle(legend_x + 9, yy, 2.3, stroke=0, fill=1)
        text(c, legend_x + 24, yy - 3, label, size=8, anchor="start")

    arrow(c, 96, 214, 96, 194, color=GRAY)
    text(c, 118, 210, "early: C0/C1 high", size=7, color=GRAY, anchor="start")
    arrow(c, 350, 214, 350, 178, color=GRAY)
    text(c, 372, 210, "late: C3/C4 high", size=7, color=GRAY, anchor="start")
    finish(c, path)


def fig_rebucket_composition() -> None:
    clusters = ["C0", "C1", "C2", "C3", "C4"]
    init_micro_math = [5, 5, 5, 5, 5]
    init_micro_zebra = [5, 5, 5, 5, 0]
    final_micro_math = [8, 2, 6, 5, 4]
    final_micro_zebra = [6, 6, 5, 3, 0]
    init_sample_math = [80, 80, 80, 80, 80]
    init_sample_zebra = [100, 100, 100, 100, 0]
    final_sample_math = [128, 32, 96, 80, 64]
    final_sample_zebra = [120, 120, 100, 60, 0]

    c, path = new_canvas("fig_rebucket_composition", 610, 390)
    text(c, 305, 368, "Experiment 1: composition drift under difficulty-label initialization", size=11, bold=True)
    text(c, 305, 352, "No accuracy calibration; initial buckets use only dataset difficulty levels", size=8, color=GRAY)

    def draw_panel(x0, y0, w, h, title, math_vals, zebra_vals, ymax, ylabel):
        draw_axes(c, x0, y0, w, h, ymax, ylabel)
        text(c, x0 + w / 2, y0 + h + 22, title, size=9, bold=True)
        slot = w / len(clusters)
        bar_w = 18
        for i, cl in enumerate(clusters):
            bx = x0 + slot * (i + 0.5) - bar_w / 2
            mh = h * math_vals[i] / ymax
            zh = h * zebra_vals[i] / ymax
            c.setFillColor(BLUE)
            c.rect(bx, y0, bar_w, mh, stroke=0, fill=1)
            c.setFillColor(RED)
            c.rect(bx, y0 + mh, bar_w, zh, stroke=0, fill=1)
            total = math_vals[i] + zebra_vals[i]
            text(c, bx + bar_w / 2, y0 + mh + zh + 5, str(total), size=6)
            text(c, bx + bar_w / 2, y0 - 12, cl, size=7)

    draw_panel(50, 210, 220, 92, "Initial micro-buckets", init_micro_math, init_micro_zebra, 15, "Micro-buckets")
    draw_panel(340, 210, 220, 92, "Final micro-buckets", final_micro_math, final_micro_zebra, 15, "")
    draw_panel(50, 48, 220, 105, "Initial samples", init_sample_math, init_sample_zebra, 265, "Samples")
    draw_panel(340, 48, 220, 105, "Final samples", final_sample_math, final_sample_zebra, 265, "")

    c.setFillColor(BLUE)
    c.rect(440, 330, 9, 9, stroke=0, fill=1)
    text(c, 453, 331, "Math", size=8, anchor="start")
    c.setFillColor(RED)
    c.rect(492, 330, 9, 9, stroke=0, fill=1)
    text(c, 505, 331, "Zebra", size=8, anchor="start")

    text(c, 305, 28, "C1 ends with 2 Math micro-buckets vs. 6 Zebra micro-buckets; C4 is Math-only both before and after.", size=7, color=DARK)
    finish(c, path)


def fig_inferred_transition_matrix() -> None:
    clusters = ["C0", "C1", "C2", "C3", "C4"]
    # Minimal adjacent-left-flow decomposition from initial/final micro-bucket counts.
    # Rows are initial clusters; columns are final clusters.
    math = [
        [5, 0, 0, 0, 0],
        [3, 2, 0, 0, 0],
        [0, 0, 5, 0, 0],
        [0, 0, 1, 4, 0],
        [0, 0, 0, 1, 4],
    ]
    zebra = [
        [5, 0, 0, 0, 0],
        [1, 4, 0, 0, 0],
        [0, 2, 3, 0, 0],
        [0, 0, 2, 3, 0],
        [0, 0, 0, 0, 0],
    ]

    c, path = new_canvas("fig_inferred_transition_matrix", 610, 285)
    text(c, 305, 265, "Inferred net transitions from micro-bucket drift", size=11, bold=True)
    text(c, 305, 250, "Rows: initial cluster; columns: final cluster. Off-diagonal mass is adjacent and toward lower-index buckets.", size=7, color=GRAY)

    def draw_matrix(x0, title, matrix, accent):
        cell = 24
        y0 = 72
        text(c, x0 + cell * 3, 224, title, size=10, bold=True)
        for j, cl in enumerate(clusters):
            text(c, x0 + (j + 1.5) * cell, y0 + 5.7 * cell, cl, size=7)
        for i, cl in enumerate(clusters):
            text(c, x0 + 0.45 * cell, y0 + (4.5 - i) * cell, cl, size=7)
        text(c, x0 - 10, y0 + cell * 2.4, "from", size=7, color=GRAY)
        for i in range(5):
            for j in range(5):
                val = matrix[i][j]
                x = x0 + (j + 1) * cell
                y = y0 + (4 - i) * cell
                if val == 0:
                    fill = colors.HexColor("#FFFFFF")
                elif i == j:
                    fill = colors.HexColor("#E5E7EB")
                else:
                    fill = accent
                c.setFillColor(fill)
                c.setStrokeColor(colors.HexColor("#9CA3AF"))
                c.rect(x, y, cell, cell, stroke=1, fill=1)
                if val:
                    text(c, x + cell / 2, y + cell / 2 - 3, str(val), size=8, bold=(i != j), color=DARK)
        text(c, x0 + cell * 3, y0 - 18, "micro-bucket count", size=7, color=GRAY)

    draw_matrix(70, "Math", math, colors.HexColor("#DBEAFE"))
    draw_matrix(360, "Zebra", zebra, colors.HexColor("#FEE2E2"))
    text(c, 305, 34, "Examples: Math C1 -> C0 = 3; Zebra C3 -> C2 = 2. This is inferred from aggregate composition, not raw event logs.", size=7, color=DARK)
    finish(c, path)


def fig_toy_rebucket_guardrails() -> None:
    data = json.loads((ROOT / "artifacts/toy_rebucket_summary.json").read_text())
    scenarios = data["scenarios"]
    titles = {
        "A_normal_drifting": "Normal drift",
        "B_persistent_outlier": "Persistent outlier",
        "C_single_spike": "Single-step spike",
    }

    c, path = new_canvas("fig_toy_rebucket_guardrails", 610, 230)
    text(c, 305, 211, "Toy re-bucketing guardrails", size=11, bold=True)
    panel_w = 170
    y_min, y_max = 0.32, 0.75
    for p, scenario in enumerate(scenarios):
        x0 = 45 + p * 190
        y0, w, h = 45, panel_w, 125
        c.setStrokeColor(DARK)
        c.line(x0, y0, x0, y0 + h)
        c.line(x0, y0, x0 + w, y0)
        c.setStrokeColor(colors.HexColor("#D1D5DB"))
        for t in range(0, 5):
            yy = y0 + h * t / 4
            c.line(x0, yy, x0 + w, yy)
        text(c, x0 + w / 2, 184, titles.get(scenario["name"], scenario["name"]), size=9, bold=True)
        steps = scenario["steps"]
        xs = [x0 + (s - min(steps)) / (max(steps) - min(steps)) * w for s in steps]
        for series, col in [(scenario["medium_mean"], BLUE), (scenario["tracked"], ORANGE)]:
            pts = [(xs[i], y0 + (series[i] - y_min) / (y_max - y_min) * h) for i in range(len(steps))]
            c.setStrokeColor(col)
            c.setFillColor(col)
            for i in range(len(pts) - 1):
                c.line(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])
            for x, y in pts:
                c.circle(x, y, 2.4, stroke=0, fill=1)
        for ms in scenario["migration_steps"]:
            xx = x0 + (ms - min(steps)) / (max(steps) - min(steps)) * w
            c.setStrokeColor(DARK_RED)
            c.setDash(3, 2)
            c.line(xx, y0, xx, y0 + h)
            c.setDash()
            text(c, xx, y0 + h + 6, "migrate", size=7, color=DARK_RED)
        for s, xx in zip(steps, xs):
            text(c, xx, y0 - 14, str(s), size=7)
    text(c, 23, 108, "State", size=8)
    c.setFillColor(BLUE)
    c.circle(438, 199, 3, stroke=0, fill=1)
    text(c, 446, 196, "Cluster mean", size=8, anchor="start")
    c.setFillColor(ORANGE)
    c.circle(520, 199, 3, stroke=0, fill=1)
    text(c, 528, 196, "Tracked sample", size=8, anchor="start")
    finish(c, path)


def main() -> None:
    fig_method_pipeline()
    fig_step200_math_zebra()
    fig_initial_accuracy_profile()
    fig_ucb_score_drift()
    fig_rebucket_composition()
    fig_inferred_transition_matrix()
    fig_toy_rebucket_guardrails()


if __name__ == "__main__":
    main()

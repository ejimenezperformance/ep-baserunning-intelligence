"""
model/make_charts_en.py
------------------------
EP Base Running Intelligence -- English versions of the chapter 1-3 charts.

The chapter charts were originally drawn with Spanish labels. This script
redraws them in English, same design (EP brand colors), WITHOUT retraining
any model: chapters 2 and 3 read their numbers from the metrics JSON files
already saved by the chapter scripts, so the values are identical to the
Spanish charts.

Chapter 1 comparison chart: no script in this repo generates
`chart_baserunning_comparativo.png`, so its three values are taken from the
published chapter 1 results table in the README (0.41 / 0.48 / 0.04, with
n = 227 / 298 / 406). If you regenerate chapter 1 with a new script, update
CH1_RESULTS below.

Outputs (report/):
  chart_baserunning_comparativo_en.png
  chart_baserunning_ch2_phases_en.png
  chart_baserunning_ch3_position_en.png

Run with: python3 model/make_charts_en.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "report"
MODEL_DIR = ROOT / "model"
REPORT_DIR.mkdir(exist_ok=True)

NAVY = "#0B1B33"
GOLD = "#D4A53A"
OFFWHITE = "#F5F3EC"
GREY = "#C8C2B4"
TEAL = "#3E7C8A"

# Chapter 1 published results (README table). Label, n, R2, bar color.
CH1_RESULTS = [
    ("Baserunning\nRun Value\n(overall)", 227, 0.41, NAVY),
    ("Extra bases\ntaken", 298, 0.48, GOLD),
    ("Successful\nstolen bases", 406, 0.04, TEAL),
]

# Spanish labels stored in the chapter 2 metrics JSON -> English
CH2_TARGET_EN = {
    "Baserunning Run Value (total)": "Baserunning Run Value (overall)",
    "Extra bases (tasa de éxito)": "Extra bases (success rate)",
    "Robo de base (tasa de éxito)": "Stolen base (success rate)",
}
CH2_PHASE_EN = {
    "v_burst_0_10": "Burst\n(0-10 ft)",
    "v_accel_10_45": "Acceleration\n(10-45 ft)",
    "v_top_45_90": "Top speed\n(45-90 ft)",
}


def style_ax(ax):
    ax.set_facecolor(OFFWHITE)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(NAVY)


def save(fig, name):
    plt.tight_layout()
    out = REPORT_DIR / name
    plt.savefig(out, facecolor=OFFWHITE, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")
    return out


def chart_ch1():
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    fig.patch.set_facecolor(OFFWHITE)
    style_ax(ax)
    labels = [f"{lab}\nn={n}" for lab, n, _, _ in CH1_RESULTS]
    vals = [v for _, _, v, _ in CH1_RESULTS]
    colors = [c for _, _, _, c in CH1_RESULTS]
    ax.bar(labels, vals, color=colors, width=0.52, zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.012, f"{v:.2f}", ha="center", fontsize=16, color=NAVY, fontweight="bold")
    ax.set_ylim(0, 0.62)
    ax.set_ylabel("R² (10-fold CV, sprint speed + home-to-first)", fontsize=11, color=NAVY)
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    ax.set_title("EP Base Running Intelligence — Chapter 1\nDoes pure speed predict real baserunning value?",
                 fontsize=13, fontweight="bold", color=NAVY, pad=14)
    return save(fig, "chart_baserunning_comparativo_en.png")


def chart_ch2():
    results = json.load(open(MODEL_DIR / "baserunning_model_ch2_metrics.json"))
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), dpi=200, sharey=True)
    fig.patch.set_facecolor(OFFWHITE)
    for ax, (es_label, en_label) in zip(axes, CH2_TARGET_EN.items()):
        rows = [r for r in results if r["label"] == es_label and len(r["features"]) == 1]
        r2s = [r["cv_r2"] for r in rows]
        labels = [CH2_PHASE_EN[r["features"][0]] for r in rows]
        colors = [GOLD if v == max(r2s) else NAVY for v in r2s]
        style_ax(ax)
        ax.bar(labels, r2s, color=colors, edgecolor=NAVY, linewidth=0.8, zorder=3)
        ax.set_title(en_label, fontsize=10.5, fontweight="bold", color=NAVY, pad=10)
        ax.set_ylim(0, max(0.5, max(r2s) * 1.25))
        ax.axhline(0, color=NAVY, linewidth=0.8)
        ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
        for i, v in enumerate(r2s):
            ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9, color=NAVY, fontweight="bold")
    axes[0].set_ylabel("R² (10-fold CV)", fontsize=11, color=NAVY)
    fig.suptitle("EP Base Running Intelligence — Chapter 2\nWhich phase of the run best explains each skill?",
                 fontsize=13, fontweight="bold", color=NAVY, y=1.04)
    return save(fig, "chart_baserunning_ch2_phases_en.png")


def chart_ch3():
    res = json.load(open(MODEL_DIR / "baserunning_model_ch3_metrics.json"))["ch1_by_position"]
    groups = ["IF", "OF", "C", "DH"]
    labels = {"IF": "Infielders\n(n={})", "OF": "Outfielders\n(n={})",
              "C": "Catchers\n(n={})", "DH": "DH\n(n={})"}
    xt = [labels[g].format(res[g]["n"]) for g in groups]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=200)
    fig.patch.set_facecolor(OFFWHITE)

    ax = axes[0]
    style_ax(ax)
    r2 = [res[g]["cv_r2"] if res[g]["regression"] else 0 for g in groups]
    colors = [GOLD if res[g]["regression"] else GREY for g in groups]
    ax.bar(xt, r2, color=colors, edgecolor=NAVY, linewidth=0.8, zorder=3)
    for i, g in enumerate(groups):
        txt = f"{r2[i]:.2f}" if res[g]["regression"] else "n too small"
        ax.text(i, max(r2[i], 0.01) + 0.01, txt, ha="center", fontsize=9, color=NAVY, fontweight="bold")
    ax.set_ylabel("R² (10-fold CV)", fontsize=11, color=NAVY)
    ax.set_title("Sprint Speed → Baserunning Value,\nby position", fontsize=11,
                 fontweight="bold", color=NAVY, pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)

    ax2 = axes[1]
    style_ax(ax2)
    speeds = [res[g]["mean_sprint_speed"] for g in groups]
    ax2.bar(xt, speeds, color=NAVY, edgecolor=NAVY, linewidth=0.8, zorder=3)
    for i, v in enumerate(speeds):
        ax2.text(i, v + 0.1, f"{v:.1f}", ha="center", fontsize=9, color=NAVY, fontweight="bold")
    ax2.set_ylabel("Average Sprint Speed (ft/sec)", fontsize=11, color=NAVY)
    ax2.set_title("Average speed,\nby position", fontsize=11, fontweight="bold", color=NAVY, pad=10)
    ax2.set_ylim(min(speeds) - 1, max(speeds) + 1)
    ax2.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)

    fig.suptitle("EP Base Running Intelligence — Chapter 3\nDoes sprint speed predict real baserunning value equally at every position?",
                 fontsize=12.5, fontweight="bold", color=NAVY, y=1.05)
    return save(fig, "chart_baserunning_ch3_position_en.png")


def main():
    chart_ch1()
    chart_ch2()
    chart_ch3()


if __name__ == "__main__":
    main()

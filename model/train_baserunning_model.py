"""
model/train_baserunning_model.py
-----------------------------------
EP Base Running Intelligence -- capítulo 1.

Question: how well does raw physical speed (sprint_speed, hp_to_1b) predict
REAL baserunning value (Savant's own runner_runs_tot), across the full
public 2026 leaderboard (n=228, no player selection)?

Data (all real, public Savant CSV exports):
  - data/sprint_speed.csv           -> sprint_speed (ft/sec), hp_to_1b (home-to-first, sec)
  - data/baserunning_run_value.csv  -> runner_runs_tot (Savant's official baserunning run value)

runner_runs_tot is normalized to a rate (per competitive run/opportunity)
to avoid confounding with playing time, same approach as the swing project's
chapter 4.

Run with: python3 model/train_baserunning_model.py
"""

import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "report"
REPORT_DIR.mkdir(exist_ok=True)
MODEL_DIR = Path(__file__).resolve().parent

NAVY = "#0B1B33"
GOLD = "#D4A53A"
OFFWHITE = "#F5F3EC"

FEATURES = ["sprint_speed", "hp_to_1b"]
TARGET = "runner_runs_per_run"


def load_and_merge():
    ss = pd.read_csv(DATA_DIR / "sprint_speed.csv", encoding="utf-8-sig")
    brv = pd.read_csv(DATA_DIR / "baserunning_run_value.csv", encoding="utf-8-sig")
    df = ss.merge(brv, on="player_id", how="inner")
    df = df.dropna(subset=FEATURES + ["runner_runs_tot", "competitive_runs"])
    df[TARGET] = df["runner_runs_tot"] / df["competitive_runs"]
    return df


def train_and_eval(df, features, target, n_splits=10):
    X = df[features].values
    y = df[target].values

    model = LinearRegression()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X, y, cv=kf)

    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    model.fit(X, y)
    coefs = dict(zip(features, model.coef_.round(5).tolist()))

    return {
        "features": features,
        "target": target,
        "n": len(df),
        "cv_folds": n_splits,
        "cv_r2": round(r2, 4),
        "cv_mae": round(mae, 5),
        "coefficients": coefs,
        "intercept": round(float(model.intercept_), 5),
    }


def make_chart(df, result):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    fig.patch.set_facecolor(OFFWHITE)
    ax.set_facecolor(OFFWHITE)

    sc = ax.scatter(df["sprint_speed"], df[TARGET], c=df["hp_to_1b"],
                     cmap="RdYlBu", s=45, edgecolors=NAVY, linewidths=0.4, zorder=3)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Home-to-First (seg, menor = mejor)", fontsize=10, color=NAVY)

    ax.set_xlabel("Sprint Speed (ft/seg)", fontsize=12, color=NAVY)
    ax.set_ylabel("Baserunning Run Value / carrera (real, Savant)", fontsize=12, color=NAVY)
    ax.set_title(f"EP Base Running Intelligence — Capítulo 1\n¿Predice la velocidad pura el valor real de corrido de bases? (n={result['n']}, R²={result['cv_r2']})",
                 fontsize=12, fontweight="bold", color=NAVY, pad=14)
    ax.grid(linestyle="--", alpha=0.3, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(NAVY)

    plt.tight_layout()
    out = REPORT_DIR / "chart_baserunning_scatter.png"
    plt.savefig(out, facecolor=OFFWHITE, bbox_inches="tight")
    plt.close()
    return out


def main():
    df = load_and_merge()
    print(f"Jugadores con sprint_speed + baserunning run value real: {len(df)}\n")

    result = train_and_eval(df, FEATURES, TARGET)
    print("=== Sprint Speed + Home-to-First -> Baserunning Run Value (real, por carrera) ===")
    print(f"  n:               {result['n']}")
    print(f"  CV R² (10-fold): {result['cv_r2']}")
    print(f"  CV MAE:          {result['cv_mae']}")
    print(f"  Coeficientes:    {result['coefficients']}")
    print()

    # Also test sprint_speed alone
    result_solo = train_and_eval(df, ["sprint_speed"], TARGET)
    print(f"  Solo sprint_speed: R²={result_solo['cv_r2']}")
    result_hp = train_and_eval(df, ["hp_to_1b"], TARGET)
    print(f"  Solo hp_to_1b:     R²={result_hp['cv_r2']}")
    print()

    out_json = MODEL_DIR / "baserunning_model_metrics.json"
    with open(out_json, "w") as f:
        json.dump([result, result_solo, result_hp], f, indent=2)
    print(f"Wrote {out_json}")

    chart_path = make_chart(df, result)
    print(f"Wrote {chart_path}")

    merged_path = DATA_DIR / "merged_baserunning_dataset_2026.csv"
    df.to_csv(merged_path, index=False)
    print(f"Wrote {merged_path}")


if __name__ == "__main__":
    main()

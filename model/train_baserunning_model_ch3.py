"""
model/train_baserunning_model_ch3.py
-------------------------------------
EP Base Running Intelligence -- capítulo 3.

Pregunta: los capítulos 1 y 2 encontraron dos patrones en la población
completa de corredores (n=227): (a) sprint speed predice bien extra bases
pero casi nada el robo de bases, y (b) la fase de aceleración (10-45 ft)
explica el valor general de corrido de bases tanto o más que la velocidad
tope. Capítulo 3 pregunta: ¿esos patrones son iguales para todas las
posiciones, o cambian según el perfil físico del jugador?

Agrupamos por posición:
  - IF (infielders): 1B, 2B, 3B, SS
  - OF (outfielders): LF, CF, RF
  - C  (catchers)
  - DH (designated hitters)

C y DH quedan con muestra chica tras el cruce con datos reales (n=11 y
n=13) -- se reportan de forma descriptiva, sin regresión, para no fingir
precisión que los datos no sostienen. IF (n=115) y OF (n=88) sí tienen
muestra suficiente para regresión con validación cruzada.

Datos (reales, públicos, Savant CSV export):
  - data/sprint_speed.csv            -> position, sprint_speed, hp_to_1b
  - data/baserunning_run_value.csv   -> runner_runs_tot
  - data/running_splits.csv          -> splits de tiempo cada 5 ft (fases, capítulo 2)

Run with: python3 model/train_baserunning_model_ch3.py
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

PHASES = ["v_burst_0_10", "v_accel_10_45", "v_top_45_90"]
MIN_N_FOR_REGRESSION = 40


def pos_group(p):
    if p == "C":
        return "C"
    if p in ("1B", "2B", "3B", "SS"):
        return "IF"
    if p in ("LF", "CF", "RF"):
        return "OF"
    return "DH"


def compute_phase_velocities(splits: pd.DataFrame) -> pd.DataFrame:
    df = splits.copy()

    def v(dist_ft, t_start_col, t_end_col):
        dt = df[t_end_col] - df[t_start_col]
        return dist_ft / dt

    df["v_burst_0_10"] = v(10, "seconds_since_hit_000", "seconds_since_hit_010")
    df["v_accel_10_45"] = v(35, "seconds_since_hit_010", "seconds_since_hit_045")
    df["v_top_45_90"] = v(45, "seconds_since_hit_045", "seconds_since_hit_090")

    keep = ["player_id", "position_name"] + PHASES
    return df[keep]


def load_ch1_style():
    """Sprint speed + hp_to_1b -> runner_runs_per_run, con grupo de posición."""
    ss = pd.read_csv(DATA_DIR / "sprint_speed.csv", encoding="utf-8-sig")
    brv = pd.read_csv(DATA_DIR / "baserunning_run_value.csv", encoding="utf-8-sig")
    df = ss.merge(brv[["player_id", "runner_runs_tot"]], on="player_id", how="inner")
    df = df.dropna(subset=["sprint_speed", "hp_to_1b", "runner_runs_tot", "competitive_runs"])
    df["runner_runs_per_run"] = df["runner_runs_tot"] / df["competitive_runs"]
    df["pos_group"] = df["position"].apply(pos_group)
    return df


def load_ch2_style():
    """Fases de velocidad -> runner_runs_tot, con grupo de posición."""
    splits_raw = pd.read_csv(DATA_DIR / "running_splits.csv", encoding="utf-8-sig")
    phases = compute_phase_velocities(splits_raw)
    brv = pd.read_csv(DATA_DIR / "baserunning_run_value.csv", encoding="utf-8-sig")
    df = phases.merge(brv[["player_id", "runner_runs_tot"]], on="player_id", how="inner")
    df = df.dropna(subset=PHASES + ["runner_runs_tot"])
    df["pos_group"] = df["position_name"].apply(pos_group)
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

    return {
        "features": features, "target": target, "n": len(df),
        "cv_r2": round(r2, 4), "cv_mae": round(mae, 5),
        "coefficients": dict(zip(features, model.coef_.round(5).tolist())),
        "intercept": round(float(model.intercept_), 5),
    }


def run_ch1_by_position(df):
    results = {}
    for grp in ["IF", "OF", "C", "DH"]:
        sub = df[df["pos_group"] == grp]
        if len(sub) >= MIN_N_FOR_REGRESSION:
            r = train_and_eval(sub, ["sprint_speed", "hp_to_1b"], "runner_runs_per_run")
            r["mean_sprint_speed"] = round(sub["sprint_speed"].mean(), 2)
            r["mean_value"] = round(sub["runner_runs_per_run"].mean(), 5)
            r["regression"] = True
        else:
            r = {
                "n": len(sub), "regression": False,
                "mean_sprint_speed": round(sub["sprint_speed"].mean(), 2),
                "mean_value": round(sub["runner_runs_per_run"].mean(), 5),
            }
        results[grp] = r
    return results


def run_ch2_by_position(df):
    results = {}
    for grp in ["IF", "OF", "C", "DH"]:
        sub = df[df["pos_group"] == grp]
        entry = {"n": len(sub)}
        for phase in PHASES:
            entry[f"mean_{phase}"] = round(sub[phase].mean(), 2)
        if len(sub) >= MIN_N_FOR_REGRESSION:
            entry["regression"] = True
            for phase in PHASES:
                r = train_and_eval(sub, [phase], "runner_runs_tot")
                entry[f"r2_{phase}"] = r["cv_r2"]
        else:
            entry["regression"] = False
        results[grp] = entry
    return results


def make_chart_ch1(results):
    groups = ["IF", "OF", "C", "DH"]
    labels = {"IF": "Infielders\n(n={})", "OF": "Outfielders\n(n={})", "C": "Catchers\n(n={})", "DH": "DH\n(n={})"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=200)
    fig.patch.set_facecolor(OFFWHITE)

    # Panel 1: R² por posición (solo IF/OF tienen regresión)
    ax = axes[0]
    ax.set_facecolor(OFFWHITE)
    r2_vals = [results[g]["cv_r2"] if results[g]["regression"] else 0 for g in groups]
    colors = [GOLD if results[g]["regression"] else "#C8C2B4" for g in groups]
    bars = ax.bar([labels[g].format(results[g]["n"]) for g in groups], r2_vals, color=colors, edgecolor=NAVY, linewidth=0.8, zorder=3)
    for i, g in enumerate(groups):
        txt = f"{r2_vals[i]:.2f}" if results[g]["regression"] else "n insuf."
        ax.text(i, max(r2_vals[i], 0.01) + 0.01, txt, ha="center", fontsize=9, color=NAVY, fontweight="bold")
    ax.set_ylabel("R² (10-fold CV)", fontsize=11, color=NAVY)
    ax.set_title("Sprint Speed → Baserunning Value,\npor posición", fontsize=11, fontweight="bold", color=NAVY, pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(NAVY)

    # Panel 2: velocidad promedio por posición
    ax2 = axes[1]
    ax2.set_facecolor(OFFWHITE)
    speeds = [results[g]["mean_sprint_speed"] for g in groups]
    ax2.bar([labels[g].format(results[g]["n"]) for g in groups], speeds, color=NAVY, edgecolor=NAVY, linewidth=0.8, zorder=3)
    for i, v in enumerate(speeds):
        ax2.text(i, v + 0.1, f"{v:.1f}", ha="center", fontsize=9, color=NAVY, fontweight="bold")
    ax2.set_ylabel("Sprint Speed promedio (ft/seg)", fontsize=11, color=NAVY)
    ax2.set_title("Velocidad promedio,\npor posición", fontsize=11, fontweight="bold", color=NAVY, pad=10)
    ax2.set_ylim(min(speeds) - 1, max(speeds) + 1)
    ax2.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax2.spines[spine].set_color(NAVY)

    fig.suptitle("EP Base Running Intelligence — Capítulo 3\n¿Sprint speed predice el valor real de corrido de bases igual en todas las posiciones?",
                 fontsize=12.5, fontweight="bold", color=NAVY, y=1.05)

    plt.tight_layout()
    out = REPORT_DIR / "chart_baserunning_ch3_position.png"
    plt.savefig(out, facecolor=OFFWHITE, bbox_inches="tight")
    plt.close()
    return out


def main():
    df1 = load_ch1_style()
    df2 = load_ch2_style()

    print(f"Capítulo 1 (sprint speed): n={len(df1)}, por posición: {df1['pos_group'].value_counts().to_dict()}\n")
    print(f"Capítulo 2 (fases): n={len(df2)}, por posición: {df2['pos_group'].value_counts().to_dict()}\n")

    results_ch1 = run_ch1_by_position(df1)
    print("=== Sprint Speed + Home-to-First -> Baserunning Run Value, por posición ===")
    for g, r in results_ch1.items():
        if r["regression"]:
            print(f"  {g}: n={r['n']:3d}  R²={r['cv_r2']}  vel.prom={r['mean_sprint_speed']} ft/s  valor.prom={r['mean_value']}")
        else:
            print(f"  {g}: n={r['n']:3d}  (muestra insuficiente para regresión)  vel.prom={r['mean_sprint_speed']} ft/s  valor.prom={r['mean_value']}")
    print()

    results_ch2 = run_ch2_by_position(df2)
    print("=== Fases de velocidad -> Baserunning Run Value (total), por posición ===")
    for g, r in results_ch2.items():
        if r["regression"]:
            print(f"  {g}: n={r['n']:3d}  R² burst={r['r2_v_burst_0_10']}  accel={r['r2_v_accel_10_45']}  top={r['r2_v_top_45_90']}")
        else:
            print(f"  {g}: n={r['n']:3d}  (muestra insuficiente para regresión)")
    print()

    out_json = MODEL_DIR / "baserunning_model_ch3_metrics.json"
    with open(out_json, "w") as f:
        json.dump({"ch1_by_position": results_ch1, "ch2_by_position": results_ch2}, f, indent=2)
    print(f"Wrote {out_json}")

    chart_path = make_chart_ch1(results_ch1)
    print(f"Wrote {chart_path}")

    df1.to_csv(DATA_DIR / "merged_baserunning_dataset_ch3_2026.csv", index=False)
    print(f"Wrote {DATA_DIR / 'merged_baserunning_dataset_ch3_2026.csv'}")


if __name__ == "__main__":
    main()

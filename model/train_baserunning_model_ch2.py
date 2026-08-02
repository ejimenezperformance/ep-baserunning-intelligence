"""
model/train_baserunning_model_ch2.py
-------------------------------------
EP Base Running Intelligence -- capítulo 2.

Pregunta: capítulo 1 encontró que sprint_speed (velocidad máxima) predice
bien "extra bases" (R²=0.48) pero casi nada el robo de base exitoso
(R²=0.04). Capítulo 2 pregunta POR QUÉ, descomponiendo la carrera en
fases usando los splits de aceleración pie por pie de Savant:

  - burst (0-10 ft):      primeros pasos, reacción/arranque
  - acceleration (10-45 ft): fase de aceleración
  - top_speed (45-90 ft): fase de velocidad máxima sostenida

Hipótesis: si el robo de base depende más del "jump" (arranque y lectura
del pitcher) que de la velocidad máxima, el burst debería predecir robo
de base mejor que el top_speed -- lo opuesto a lo que pasa con extra
bases, donde domina la velocidad pura.

Datos (reales, públicos, Savant CSV export):
  - data/running_splits.csv         -> tiempo (seg) en cada 5 ft desde el contacto, 0-90 ft
  - data/baserunning_run_value.csv  -> runner_runs_tot (valor total real)
  - data/base_running.csv           -> n_att_xb / rate_att_xb (extra bases)
  - data/basestealing_running_game.csv -> n_sb, n_cs (robo de base)

Run with: python3 model/train_baserunning_model_ch2.py
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


def compute_phase_velocities(splits: pd.DataFrame) -> pd.DataFrame:
    """Convierte los tiempos acumulados en velocidad promedio (ft/seg) por fase."""
    df = splits.copy()

    def v(dist_ft, t_start_col, t_end_col):
        dt = df[t_end_col] - df[t_start_col]
        return dist_ft / dt

    # Burst: primeros 10 ft (arranque desde el home / la base)
    df["v_burst_0_10"] = v(10, "seconds_since_hit_000", "seconds_since_hit_010")
    # Aceleración: de 10 a 45 ft
    df["v_accel_10_45"] = v(35, "seconds_since_hit_010", "seconds_since_hit_045")
    # Velocidad tope: de 45 a 90 ft (fase donde ya alcanzó velocidad de crucero)
    df["v_top_45_90"] = v(45, "seconds_since_hit_045", "seconds_since_hit_090")

    keep = ["player_id", "position_name", "age"] + PHASES
    return df[keep]


def load_and_merge():
    splits_raw = pd.read_csv(DATA_DIR / "running_splits.csv", encoding="utf-8-sig")
    phases = compute_phase_velocities(splits_raw)

    brv = pd.read_csv(DATA_DIR / "baserunning_run_value.csv", encoding="utf-8-sig")
    xb = pd.read_csv(DATA_DIR / "base_running.csv", encoding="utf-8-sig")
    xb = xb.rename(columns={"entity_id": "player_id"})
    sb = pd.read_csv(DATA_DIR / "basestealing_running_game.csv", encoding="utf-8-sig")

    df = phases.merge(brv[["player_id", "runner_runs_tot"]], on="player_id", how="inner")
    df = df.merge(
        xb[["player_id", "rate_att_xb", "rate_safe"]].drop_duplicates("player_id"),
        on="player_id", how="inner",
    )
    df = df.merge(
        sb[["player_id", "n_sb", "n_cs"]].drop_duplicates("player_id"),
        on="player_id", how="inner",
    )

    df = df.dropna(subset=PHASES + ["runner_runs_tot", "rate_safe"])
    df["sb_attempts"] = df["n_sb"] + df["n_cs"]
    df = df[df["sb_attempts"] >= 3].copy()  # muestra mínima para tasa de éxito confiable
    df["sb_success_rate"] = df["n_sb"] / df["sb_attempts"]

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


def run_all_phase_tests(df):
    """Para cada skill, testea cada fase por separado y las 3 juntas."""
    targets = {
        "runner_runs_tot": "Baserunning Run Value (total)",
        "rate_safe": "Extra bases (tasa de éxito)",
        "sb_success_rate": "Robo de base (tasa de éxito)",
    }

    results = []
    for target, label in targets.items():
        for phase in PHASES:
            r = train_and_eval(df, [phase], target)
            r["label"] = label
            results.append(r)
        r_all = train_and_eval(df, PHASES, target)
        r_all["label"] = label
        results.append(r_all)

    return results


def make_chart(results):
    targets = ["Baserunning Run Value (total)", "Extra bases (tasa de éxito)", "Robo de base (tasa de éxito)"]
    phase_labels = {"v_burst_0_10": "Burst\n(0-10 ft)", "v_accel_10_45": "Aceleración\n(10-45 ft)", "v_top_45_90": "Vel. tope\n(45-90 ft)"}

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), dpi=200, sharey=True)
    fig.patch.set_facecolor(OFFWHITE)

    for ax, target_label in zip(axes, targets):
        rows = [r for r in results if r["label"] == target_label and len(r["features"]) == 1]
        r2s = [r["cv_r2"] for r in rows]
        labels = [phase_labels[r["features"][0]] for r in rows]
        colors = [GOLD if v == max(r2s) else NAVY for v in r2s]

        ax.set_facecolor(OFFWHITE)
        ax.bar(labels, r2s, color=colors, edgecolor=NAVY, linewidth=0.8, zorder=3)
        ax.set_title(target_label, fontsize=10.5, fontweight="bold", color=NAVY, pad=10)
        ax.set_ylim(0, max(0.5, max(r2s) * 1.25))
        ax.axhline(0, color=NAVY, linewidth=0.8)
        ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(NAVY)
        for i, v in enumerate(r2s):
            ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9, color=NAVY, fontweight="bold")

    axes[0].set_ylabel("R² (10-fold CV)", fontsize=11, color=NAVY)
    fig.suptitle("EP Base Running Intelligence — Capítulo 2\n¿Qué fase de la carrera explica mejor cada habilidad?",
                 fontsize=13, fontweight="bold", color=NAVY, y=1.04)

    plt.tight_layout()
    out = REPORT_DIR / "chart_baserunning_ch2_phases.png"
    plt.savefig(out, facecolor=OFFWHITE, bbox_inches="tight")
    plt.close()
    return out


def main():
    df = load_and_merge()
    print(f"Jugadores con splits + las 3 métricas de valor real: {len(df)}\n")

    results = run_all_phase_tests(df)

    for r in results:
        feat_str = "+".join(r["features"])
        print(f"[{r['label']:35s}] {feat_str:30s} n={r['n']:3d}  R²={r['cv_r2']}")

    out_json = MODEL_DIR / "baserunning_model_ch2_metrics.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_json}")

    chart_path = make_chart(results)
    print(f"Wrote {chart_path}")

    merged_path = DATA_DIR / "merged_baserunning_dataset_ch2_2026.csv"
    df.to_csv(merged_path, index=False)
    print(f"Wrote {merged_path}")


if __name__ == "__main__":
    main()

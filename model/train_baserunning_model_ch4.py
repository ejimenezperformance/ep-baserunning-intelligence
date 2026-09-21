"""
model/train_baserunning_model_ch4.py
-------------------------------------
EP Base Running Intelligence -- capítulo 4: "Decisión, no velocidad".

Preguntas:
  A. Los capítulos 1-3 midieron cuánto explica la velocidad del VALOR de
     corrido de bases. Pero un corredor decide dos cosas distintas al
     avanzar una base extra: (1) SI intenta y (2) SI sale safe cuando
     intenta. ¿La velocidad predice por igual esas dos cosas?
  B. ¿Cómo se relacionan el lead primario y secundario con la frecuencia
     de intentos de robo y con su éxito, una vez descontada la velocidad?

Datos (reales, públicos, Savant CSV export, temporada 2026):
  - data/sprint_speed.csv                -> sprint_speed, hp_to_1b
  - data/base_running.csv                -> n_opp_xb, n_att_xb, rate_att_xb,
                                            n_safe, n_out, rate_safe_per_attempt
  - data/basestealing_running_game.csv   -> n_init, n_sb, n_cs, rate_sbx,
                                            r_primary_lead, r_secondary_lead

Notas de definición (verificadas contra los datos):
  - rate_sbx == (n_sb + n_cs) / n_init  (tasa de INTENTO de robo por
    oportunidad, no tasa de éxito). Se verificó fila por fila.
  - Extra bases: rate_att_xb = intentos / oportunidades.

Sobre la interpretación: el lead es en parte una decisión del propio
corredor (quien planea correr suele alejarse más), así que las
asociaciones del bloque B NO prueban que un lead mayor cause más robos.

Run with: python3 model/train_baserunning_model_ch4.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "report"
REPORT_DIR.mkdir(exist_ok=True)
MODEL_DIR = Path(__file__).resolve().parent

NAVY = "#0B1B33"
GOLD = "#D4A53A"
OFFWHITE = "#F5F3EC"
GREY = "#C8C2B4"

MIN_OPP_XB = 30          # oportunidades mínimas de extra base
MIN_ATT_XB = 10          # intentos mínimos para medir éxito por intento
MIN_ATT_SENSITIVITY = [5, 10, 15, 20]
MIN_ATT_STEAL = 8        # intentos de robo mínimos para medir % de éxito
SEED = 42


# --------------------------------------------------------------------------
# Utilidades estadísticas
# --------------------------------------------------------------------------

def cv_r2(df, features, target, n_splits=10):
    """R² con validación cruzada de 10 folds (mismo estándar que cap. 1-3)."""
    X = df[features].values
    y = df[target].values
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    pred = cross_val_predict(LinearRegression(), X, y, cv=kf)
    return float(r2_score(y, pred))


def pearson(df, a, b):
    r, p = stats.pearsonr(df[a], df[b])
    return {"n": int(len(df)), "r": round(float(r), 4), "p": float(f"{p:.3g}")}


def bootstrap_ci_r2(df, features, target, n_boot=500):
    """IC 95% bootstrap del R² CV (para no sobre-interpretar diferencias chicas)."""
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(df), len(df))
        sub = df.iloc[idx]
        vals.append(cv_r2(sub, features, target))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [round(float(lo), 3), round(float(hi), 3)]


# --------------------------------------------------------------------------
# Carga
# --------------------------------------------------------------------------

def load_speed():
    ss = pd.read_csv(DATA_DIR / "sprint_speed.csv", encoding="utf-8-sig")
    return ss[["player_id", "position", "sprint_speed", "hp_to_1b"]]


def load_xb():
    br = pd.read_csv(DATA_DIR / "base_running.csv", encoding="utf-8-sig")
    br = br.rename(columns={"entity_id": "player_id"})
    keep = ["player_id", "entity_name", "n_opp_xb", "n_att_xb", "rate_att_xb",
            "n_safe", "n_out", "rate_safe_per_attempt", "runner_runs_advances"]
    return br[keep]


def load_steals():
    bs = pd.read_csv(DATA_DIR / "basestealing_running_game.csv", encoding="utf-8-sig")
    bs = bs[bs["key_target_base"] == "All"].copy()
    for c in ["r_primary_lead", "r_secondary_lead", "r_sec_minus_prim_lead"]:
        bs[c] = pd.to_numeric(bs[c], errors="coerce")
    bs["n_att_steal"] = bs["n_sb"] + bs["n_cs"]
    bs["steal_success_rate"] = np.where(bs["n_att_steal"] > 0,
                                        bs["n_sb"] / bs["n_att_steal"].replace(0, np.nan),
                                        np.nan)
    keep = ["player_id", "player_name", "n_init", "n_sb", "n_cs", "n_att_steal",
            "rate_sbx", "steal_success_rate", "runs_stolen_on_running_act",
            "r_primary_lead", "r_secondary_lead", "r_sec_minus_prim_lead"]
    return bs[keep].dropna(subset=["r_primary_lead", "r_secondary_lead"])


# --------------------------------------------------------------------------
# Bloque A: velocidad -> intentar vs. salir safe (extra bases)
# --------------------------------------------------------------------------

def run_block_a(speed, xb):
    df = speed.merge(xb, on="player_id", how="inner")
    df = df.dropna(subset=["sprint_speed", "rate_att_xb"])
    df = df[df["n_opp_xb"] >= MIN_OPP_XB]

    out = {"filters": {"min_opportunities": MIN_OPP_XB, "min_attempts_for_success": MIN_ATT_XB},
           "n_att_population": int(len(df))}

    # Población completa para "intentar"
    out["attempt_rate_all"] = {
        "cv_r2": round(cv_r2(df, ["sprint_speed"], "rate_att_xb"), 4),
        "pearson": pearson(df, "sprint_speed", "rate_att_xb"),
    }

    # Comparación justa: MISMA muestra (>= MIN_ATT_XB intentos) para ambas variables
    same = df[df["n_att_xb"] >= MIN_ATT_XB].dropna(subset=["rate_safe_per_attempt"])
    attempt_cols = ["sprint_speed"]
    out["same_sample"] = {
        "n": int(len(same)),
        "attempt_rate": {
            "cv_r2": round(cv_r2(same, attempt_cols, "rate_att_xb"), 4),
            "ci95": bootstrap_ci_r2(same, attempt_cols, "rate_att_xb"),
            "pearson": pearson(same, "sprint_speed", "rate_att_xb"),
        },
        "safe_per_attempt": {
            "cv_r2": round(cv_r2(same, attempt_cols, "rate_safe_per_attempt"), 4),
            "ci95": bootstrap_ci_r2(same, attempt_cols, "rate_safe_per_attempt"),
            "pearson": pearson(same, "sprint_speed", "rate_safe_per_attempt"),
        },
        "mean_safe_per_attempt": round(float(same["rate_safe_per_attempt"].mean()), 4),
        "sd_safe_per_attempt": round(float(same["rate_safe_per_attempt"].std()), 4),
    }

    # Sensibilidad al umbral de intentos mínimos
    sens = {}
    for k in MIN_ATT_SENSITIVITY:
        s = df[df["n_att_xb"] >= k].dropna(subset=["rate_safe_per_attempt"])
        if len(s) < 40:
            continue
        sens[str(k)] = {
            "n": int(len(s)),
            "r2_attempt": round(cv_r2(s, attempt_cols, "rate_att_xb"), 4),
            "r2_safe_per_attempt": round(cv_r2(s, attempt_cols, "rate_safe_per_attempt"), 4),
        }
    out["sensitivity_min_attempts"] = sens
    return out, same


# --------------------------------------------------------------------------
# Bloque B: leads vs. robo
# --------------------------------------------------------------------------

def run_block_b(speed, steals):
    df = steals.merge(speed[["player_id", "sprint_speed"]], on="player_id", how="inner")
    df = df.dropna(subset=["sprint_speed"])
    out = {"n_population": int(len(df))}

    # ¿Los corredores más rápidos se alejan más?
    out["speed_vs_primary_lead"] = pearson(df, "sprint_speed", "r_primary_lead")
    out["speed_vs_secondary_lead"] = pearson(df, "sprint_speed", "r_secondary_lead")

    # Lead vs frecuencia de intento (rate_sbx = intentos / oportunidades)
    out["primary_lead_vs_attempt_rate"] = pearson(df, "r_primary_lead", "rate_sbx")
    out["secondary_lead_vs_attempt_rate"] = pearson(df, "r_secondary_lead", "rate_sbx")

    # ¿El lead agrega algo por encima de la velocidad?
    base_feats = ["sprint_speed"]
    feats_p = ["sprint_speed", "r_primary_lead"]
    feats_ps = ["sprint_speed", "r_primary_lead", "r_secondary_lead"]
    out["attempt_rate_models"] = {
        "speed_only": {"cv_r2": round(cv_r2(df, base_feats, "rate_sbx"), 4),
                       "ci95": bootstrap_ci_r2(df, base_feats, "rate_sbx")},
        "speed_plus_primary_lead": {"cv_r2": round(cv_r2(df, feats_p, "rate_sbx"), 4),
                                    "ci95": bootstrap_ci_r2(df, feats_p, "rate_sbx")},
        "speed_plus_both_leads": {"cv_r2": round(cv_r2(df, feats_ps, "rate_sbx"), 4),
                                  "ci95": bootstrap_ci_r2(df, feats_ps, "rate_sbx")},
    }

    # Lead vs éxito del robo (solo jugadores con intentos suficientes)
    succ = df[df["n_att_steal"] >= MIN_ATT_STEAL].dropna(subset=["steal_success_rate"])
    out["success_sample"] = {
        "min_attempts": MIN_ATT_STEAL, "n": int(len(succ)),
        "mean_success_rate": round(float(succ["steal_success_rate"].mean()), 4),
        "primary_lead_vs_success": pearson(succ, "r_primary_lead", "steal_success_rate"),
        "secondary_lead_vs_success": pearson(succ, "r_secondary_lead", "steal_success_rate"),
        "speed_vs_success": pearson(succ, "sprint_speed", "steal_success_rate"),
    }

    # Lead vs valor de carreras del robo (entre quienes intentan lo suficiente)
    out["primary_lead_vs_runs_stolen"] = pearson(df, "r_primary_lead", "runs_stolen_on_running_act")
    return out, df, succ


# --------------------------------------------------------------------------
# Gráfico
# --------------------------------------------------------------------------

def style_ax(ax):
    ax.set_facecolor(OFFWHITE)
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(NAVY)


TEXT = {
    "es": {
        "labels": ["¿Intenta la\nbase extra?", "¿Sale safe\ncuando intenta?"],
        "ylabel1": "R² (10-fold CV) de Sprint Speed",
        "title1": "Velocidad → decisión de avanzar\n(extra bases, n={n}, barras = IC 95% bootstrap)",
        "xlabel2": "Lead primario (ft)",
        "ylabel2": "Intentos de robo por oportunidad (%)",
        "title2": "Lead primario vs. frecuencia de intento de robo\n(asociación, no causalidad)",
        "suptitle": "EP Base Running Intelligence — Capítulo 4\nDecisión, no velocidad",
        "file": "chart_baserunning_ch4_decision.png",
    },
    "en": {
        "labels": ["Does he attempt\nthe extra base?", "Is he safe\nwhen he attempts?"],
        "ylabel1": "Sprint Speed R² (10-fold CV)",
        "title1": "Speed → decision to advance\n(extra bases, n={n}, bars = 95% bootstrap CI)",
        "xlabel2": "Primary lead (ft)",
        "ylabel2": "Steal attempts per opportunity (%)",
        "title2": "Primary lead vs. steal attempt frequency\n(association, not causation)",
        "suptitle": "EP Base Running Intelligence — Chapter 4\nDecision, not speed",
        "file": "chart_baserunning_ch4_decision_en.png",
    },
}


def make_chart(res_a, res_b, df_b, lang="es"):
    t = TEXT[lang]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), dpi=200)
    fig.patch.set_facecolor(OFFWHITE)

    # Panel 1: velocidad -> intentar vs. salir safe
    ax = axes[0]
    style_ax(ax)
    s = res_a["same_sample"]
    labels = t["labels"]
    vals = [s["attempt_rate"]["cv_r2"], max(s["safe_per_attempt"]["cv_r2"], 0)]
    cis = [s["attempt_rate"]["ci95"], s["safe_per_attempt"]["ci95"]]
    ax.bar(labels, vals, color=[GOLD, GREY], edgecolor=NAVY, linewidth=0.8, zorder=3, width=0.55)
    for i, (v, ci) in enumerate(zip(vals, cis)):
        ax.errorbar(i, v, yerr=[[max(v - max(ci[0], 0), 0)], [max(ci[1] - v, 0)]],
                    color=NAVY, capsize=4, linewidth=1, zorder=4)
        ax.text(i, ci[1] + 0.02, f"R² = {v:.2f}", ha="center", fontsize=10, color=NAVY, fontweight="bold")
    ax.set_ylabel(t["ylabel1"], fontsize=11, color=NAVY)
    ax.set_ylim(0, max(c[1] for c in cis) + 0.12)
    ax.set_title(t["title1"].format(n=s["n"]), fontsize=11, fontweight="bold", color=NAVY, pad=10)

    # Panel 2: lead primario vs frecuencia de intento de robo
    ax2 = axes[1]
    style_ax(ax2)
    ax2.grid(axis="x", linestyle="--", alpha=0.3, zorder=0)
    x = df_b["r_primary_lead"].values
    y = df_b["rate_sbx"].values * 100
    ax2.scatter(x, y, s=16, color=NAVY, alpha=0.55, edgecolor="none", zorder=3)
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax2.plot(xs, slope * xs + intercept, color=GOLD, linewidth=2.2, zorder=4)
    r = res_b["primary_lead_vs_attempt_rate"]["r"]
    ax2.text(0.04, 0.93, f"r = {r:.2f}  (n={res_b['n_population']})", transform=ax2.transAxes,
             fontsize=10, color=NAVY, fontweight="bold", va="top")
    ax2.set_xlabel(t["xlabel2"], fontsize=11, color=NAVY)
    ax2.set_ylabel(t["ylabel2"], fontsize=11, color=NAVY)
    ax2.set_title(t["title2"], fontsize=11, fontweight="bold", color=NAVY, pad=10)

    fig.suptitle(t["suptitle"], fontsize=12.5, fontweight="bold", color=NAVY, y=1.06)
    plt.tight_layout()
    out = REPORT_DIR / t["file"]
    plt.savefig(out, facecolor=OFFWHITE, bbox_inches="tight")
    plt.close()
    return out


# --------------------------------------------------------------------------

def main():
    speed = load_speed()
    xb = load_xb()
    steals = load_steals()

    res_a, same_a = run_block_a(speed, xb)
    res_b, df_b, succ_b = run_block_b(speed, steals)

    s = res_a["same_sample"]
    print("=== A. Extra bases: velocidad -> intentar vs. salir safe ===")
    print(f"  Población (>= {MIN_OPP_XB} oportunidades): n={res_a['n_att_population']}  "
          f"R² intento={res_a['attempt_rate_all']['cv_r2']}")
    print(f"  Misma muestra (>= {MIN_ATT_XB} intentos), n={s['n']}:")
    print(f"    R² velocidad -> tasa de intento:       {s['attempt_rate']['cv_r2']}  IC95 {s['attempt_rate']['ci95']}  r={s['attempt_rate']['pearson']['r']} p={s['attempt_rate']['pearson']['p']}")
    print(f"    R² velocidad -> safe por intento:      {s['safe_per_attempt']['cv_r2']}  IC95 {s['safe_per_attempt']['ci95']}  r={s['safe_per_attempt']['pearson']['r']} p={s['safe_per_attempt']['pearson']['p']}")
    print(f"  Sensibilidad al umbral de intentos: {res_a['sensitivity_min_attempts']}\n")

    print("=== B. Leads y robo ===")
    print(f"  n={res_b['n_population']}")
    print(f"  velocidad vs lead primario:   {res_b['speed_vs_primary_lead']}")
    print(f"  velocidad vs lead secundario: {res_b['speed_vs_secondary_lead']}")
    print(f"  lead primario vs tasa de intento:   {res_b['primary_lead_vs_attempt_rate']}")
    print(f"  lead secundario vs tasa de intento: {res_b['secondary_lead_vs_attempt_rate']}")
    print(f"  Modelos de tasa de intento: {json.dumps(res_b['attempt_rate_models'])}")
    print(f"  Éxito del robo (>= {MIN_ATT_STEAL} intentos): {json.dumps(res_b['success_sample'])}")
    print(f"  lead primario vs carreras del robo: {res_b['primary_lead_vs_runs_stolen']}\n")

    out_json = MODEL_DIR / "baserunning_model_ch4_metrics.json"
    with open(out_json, "w") as f:
        json.dump({"block_a_extra_bases": res_a, "block_b_leads_steals": res_b}, f, indent=2)
    print(f"Wrote {out_json}")

    for lang in ("es", "en"):
        chart = make_chart(res_a, res_b, df_b, lang=lang)
        print(f"Wrote {chart}")

    merged = df_b.merge(xb.drop(columns=["entity_name"]), on="player_id", how="left")
    merged.to_csv(DATA_DIR / "merged_baserunning_dataset_ch4_2026.csv", index=False)
    print(f"Wrote {DATA_DIR / 'merged_baserunning_dataset_ch4_2026.csv'}")


if __name__ == "__main__":
    main()

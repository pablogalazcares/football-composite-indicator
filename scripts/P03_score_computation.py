"""
P03 — Score Component Computation (score_OBV + score_PSR)
=========================================================
Objetivo:
    Computar los dos componentes del score por jugador-temporada:
      - score_OBV: basado en percentiles de OBV por (season, position, event_type)
      - score_PSR: basado en percentiles de PSR por (season, position, event_type)

    IMPORTANTE: este módulo NO computa score_pt = (1-α)·score_OBV + α·score_PSR.
    La decisión sobre α pertenece a P05_model_battery.py, que evalúa múltiples
    valores de α como parte de la batería de modelos. P03 solo genera los insumos.

    Además, P03 provee la función genérica `compute_score(config)` que P05 llama
    para componer score_pt con un α específico, sin re-computar los componentes.

Especificación:

  score_OBV (§5.1) — diseño post-EDA (AR1-proporcional, sign fix, shrinkage):
    1. Por cada (jugador, temporada, tipo_evento): media de OBV ya computada en P02.
    2a. Bayesian shrinkage K=10 para Shot (obv_for_mean) antes del percentil.
    2b. Percentilizar obv_for_mean dentro de (season, position, event_type) → [0,100].
    2c. Percentilizar -obv_against_mean (SIGN FIX: negativo = buena defensa) → [0,100].
    3. Promedio ponderado por AR1 por (posición, evento) — umbral ≥0.10.
       Eventos y pesos varían por posición (AR1_WEIGHTS_FOR / AR1_WEIGHTS_AGAINST).
    4. Blend macro-posicional: score_OBV = W_for·OBV_for_pt + W_against·OBV_against_pt

  score_PSR (§5.2, método consistente con OBV):
    1. PSR computado por (player, season, event_type) en P02.
    2. Percentilizar dentro de (season_name, position_group, event_type) → [0,100]
    3. Promediar percentiles por (player, season) → score_PSR

  score_pt (§5.3) — computado en P05, no aquí:
    score_pt = (1-α)·score_OBV + α·score_PSR, clipped a [0, 100]

Función genérica para P05:
    `compute_score(obv_agg, psr_agg, positions, config, psr_ev_agg)` computa
    score_pt dado un ScoreConfig con alpha. Llamada desde P05 en la batería.

Outputs:
    outputs/pipeline/p03_score_player_season.parquet
    Columnas: player_id, player_name, season_name, position_group,
              obv_for_pt, obv_against_pt, score_OBV, score_PSR, has_psr
    (NO incluye score_pt — eso lo decide P05)

Upstream:
    P02: outputs/pipeline/p02_obv_player_season.parquet
    P02: outputs/pipeline/p02_psr_player_season.parquet
    P02: outputs/pipeline/p02_psr_by_event.parquet
    P01: outputs/pipeline/p01_canonical_positions.parquet
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
OBV_AGG_PATH     = REPO / "outputs/pipeline/p02_obv_player_season.parquet"
PSR_AGG_PATH     = REPO / "outputs/pipeline/p02_psr_player_season.parquet"
PSR_EV_AGG_PATH  = REPO / "outputs/pipeline/p02_psr_by_event.parquet"
POS_PATH      = REPO / "outputs/pipeline/p01_canonical_positions.parquet"
OUT_DIR       = REPO / "outputs/pipeline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Pesos macro-posicionales (blend OBV_for / OBV_against) — paper §5 Tabla 1
MACRO_WEIGHTS: dict[str, dict[str, float]] = {
    "GK":  {"w_for": 0.10, "w_against": 0.90},
    "DEF": {"w_for": 0.25, "w_against": 0.75},
    "MID": {"w_for": 0.53, "w_against": 0.47},
    "FWD": {"w_for": 0.83, "w_against": 0.17},
}

# Diseño post-EDA (umbral AR1 ≥ 0.10, docs/experiments/eda_obv_event_types/).
# Fuente: 04_ar1_stability.csv — Spearman ρ entre percentiles t/t+1 por (posición, evento).
# Fixes respecto al diseño anterior:
#   1. SIGN FIX: obv_against se rankea con rank(-obv_against_mean) — negativo = buena defensa.
#   2. PESOS AR1-PROPORCIONALES: peso de cada evento ∝ AR1, normalizado a suma=1 por posición.
#   3. POSICIÓN-ESPECÍFICO: eventos incluidos varían por posición (GK solo para GK, etc.).
#   4. SHOT con shrinkage bayesiano K=10 para FWD/MID antes del ranking percentil.

SHOT_SHRINKAGE_K: int = 10

# Pesos FOR por posición: {event_type: AR1} — solo eventos con AR1 ≥ 0.10
# Shot FWD/MID incluido por diseño (shrinkage reduce ruido; AR1 < 0.10 pero señal posicional relevante)
AR1_WEIGHTS_FOR: dict[str, dict[str, float]] = {
    "GK": {
        "Pass":        0.433,
        "Carry":       0.430,
        "Goal Keeper": 0.209,
    },
    "DEF": {
        "Pass":           0.634,
        "Carry":          0.498,
        "Foul Committed": 0.219,
        "Clearance":      0.188,
        "Dribble":        0.179,
    },
    "MID": {
        "Pass":           0.469,
        "Carry":          0.392,
        "Dribble":        0.238,
        "Clearance":      0.220,
        "Foul Committed": 0.217,
        "Duel":           0.155,
        "Shot":           0.041,   # incluido con shrinkage bayesiano K=10
    },
    "FWD": {
        "Pass":           0.535,
        "Carry":          0.375,
        "Clearance":      0.281,
        "Dribble":        0.227,
        "Foul Committed": 0.213,
        "Duel":           0.187,
        "Shot":           0.078,   # incluido con shrinkage bayesiano K=10
    },
}

# Pesos AGAINST por posición: {event_type: AR1} — solo eventos con AR1 ≥ 0.10
# El ranking se aplica sobre -obv_against_mean (SIGN FIX): mayor percentil = mejor defensa.
AR1_WEIGHTS_AGAINST: dict[str, dict[str, float]] = {
    "GK": {
        "Goal Keeper": 0.444,
        "Carry":       0.266,
        "Pass":        0.186,
    },
    "DEF": {
        "Duel":           0.504,
        "Pass":           0.449,
        "Interception":   0.326,
        "Foul Committed": 0.266,
        "Carry":          0.252,
        "Dribble":        0.112,
    },
    "MID": {
        "Dribble":        0.345,
        "Carry":          0.319,
        "Foul Committed": 0.273,
        "Duel":           0.239,
        "Block":          0.187,
        "Interception":   0.155,
        "Clearance":      0.114,
    },
    "FWD": {
        "Foul Committed": 0.328,
        "Duel":           0.287,
        "Dribble":        0.279,
        "Clearance":      0.196,
        "Pass":           0.153,
    },
}

# PSR fuente: p02_psr_player_season.parquet (computado en P02 desde raw events)
DEFAULT_PSR_MODE: Literal["pressure_success_rate"] = "pressure_success_rate"


@dataclass
class ScoreConfig:
    """Configuración completa para un run de score_pt. Usada en batería de modelos."""
    alpha: float = 0.8
    min_minutes: int = 500          # se aplica al unir con datos de minutos
    psr_mode: str = "pressure_success_rate"
    percentile_method: str = "average"  # pandas rank method

    def as_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "min_minutes": self.min_minutes,
            "psr_mode": self.psr_mode,
            "percentile_method": self.percentile_method,
        }


# ── Helpers para el diseño AR1-proporcional ────────────────────────────────────

def _build_weight_df(ar1_table: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Convierte tabla AR1 en DataFrame (position_group, event_type, norm_weight).
    Normaliza pesos a suma=1 por posición."""
    rows = []
    for pos, ev_weights in ar1_table.items():
        total = sum(ev_weights.values())
        for et, ar1 in ev_weights.items():
            rows.append({
                "position_group": pos,
                "event_type": et,
                "norm_weight": ar1 / total,
            })
    return pd.DataFrame(rows)


def apply_bayesian_shrinkage_shot(
    df: pd.DataFrame,
    col: str = "obv_for_mean",
    K: int = SHOT_SHRINKAGE_K,
) -> pd.DataFrame:
    """
    Regularización bayesiana para obv_for_mean de eventos Shot.

    obv_shrunk = (n/(n+K)) * obv_for_mean + (K/(n+K)) * mu_grupo

    Reduce ruido para jugadores con pocos disparos shrinkando hacia la media
    del grupo (season, position_group). Solo modifica filas con event_type='Shot'.
    """
    df = df.copy()
    shot_mask = df["event_type"] == "Shot"
    if shot_mask.sum() == 0:
        return df

    group_mean = (
        df[shot_mask]
        .groupby(["season_name", "position_group"])[col]
        .transform("mean")
    )
    n = df.loc[shot_mask, "n_events"].fillna(0).clip(lower=0)
    df.loc[shot_mask, col] = (
        (n / (n + K)) * df.loc[shot_mask, col]
        + (K / (n + K)) * group_mean
    )
    return df


# ── Componente OBV ─────────────────────────────────────────────────────────────

def compute_score_obv(
    obv_agg: pd.DataFrame,
    positions: pd.DataFrame,
    percentile_method: str = "average",
) -> pd.DataFrame:
    """
    Implementa §5.1 Steps 2–4 con diseño post-EDA.

    Cambios respecto al diseño anterior:
      - SIGN FIX: obv_against rankea con -obv_against_mean (positivo = cede OBV = malo).
      - Pesos proporcionales al AR1 por (posición, evento), normalizados a suma=1.
      - Inclusión de eventos posición-específica (AR1_WEIGHTS_FOR/AGAINST).
      - Bayesian shrinkage K=10 para Shot antes del ranking percentil.

    Input: obv_agg de P02 (columnas: player_id, season_name, event_type,
           obv_for_mean, obv_against_mean, n_events, position_group)
    Output: DataFrame con player_id, season_name, score_OBV, obv_for_pt, obv_against_pt
    """
    df = obv_agg.merge(
        positions[["player_id", "position_group"]],
        on="player_id", how="left", suffixes=("", "_canon")
    )
    if "position_group_canon" in df.columns:
        df["position_group"] = df["position_group_canon"].fillna(df["position_group"])
        df = df.drop(columns=["position_group_canon"])

    # Bayesian shrinkage para Shot antes de percentilizar
    df = apply_bayesian_shrinkage_shot(df, col="obv_for_mean", K=SHOT_SHRINKAGE_K)

    # Step 2a: percentilizar obv_for_mean — mayor = mejor contribución ofensiva
    df["pct_obv_for"] = (
        df.groupby(["season_name", "position_group", "event_type"])["obv_for_mean"]
        .rank(method=percentile_method, pct=True, na_option="keep")
        * 100
    )

    # Step 2b: percentilizar -obv_against_mean (SIGN FIX)
    # obv_against < 0 → rival pierde OBV → buena defensa → mayor percentil
    df["neg_obv_against"] = -df["obv_against_mean"]
    df["pct_obv_against"] = (
        df.groupby(["season_name", "position_group", "event_type"])["neg_obv_against"]
        .rank(method=percentile_method, pct=True, na_option="keep")
        * 100
    )

    # Step 3: promedio ponderado por AR1 (posición-específico)
    w_for_df     = _build_weight_df(AR1_WEIGHTS_FOR)
    w_against_df = _build_weight_df(AR1_WEIGHTS_AGAINST)

    df_for = df.merge(
        w_for_df.rename(columns={"norm_weight": "w_for"}),
        on=["position_group", "event_type"], how="inner",
    )
    df_against = df.merge(
        w_against_df.rename(columns={"norm_weight": "w_against"}),
        on=["position_group", "event_type"], how="inner",
    )

    def _weighted_mean(sub: pd.DataFrame, pct_col: str, w_col: str) -> float:
        valid = sub[[pct_col, w_col]].dropna()
        if len(valid) == 0:
            return np.nan
        w_sum = valid[w_col].sum()
        return float((valid[pct_col] * valid[w_col]).sum() / w_sum) if w_sum > 0 else np.nan

    obv_for_pt = (
        df_for
        .groupby(["player_id", "season_name"])
        .apply(lambda x: _weighted_mean(x, "pct_obv_for", "w_for"))
        .reset_index(name="obv_for_pt")
    )
    obv_against_pt = (
        df_against
        .groupby(["player_id", "season_name"])
        .apply(lambda x: _weighted_mean(x, "pct_obv_against", "w_against"))
        .reset_index(name="obv_against_pt")
    )

    meta = (
        df.drop_duplicates(["player_id", "season_name"])
        [["player_id", "season_name", "position_group", "player_name"]]
    )
    combined = (
        meta
        .merge(obv_for_pt,     on=["player_id", "season_name"], how="left")
        .merge(obv_against_pt, on=["player_id", "season_name"], how="left")
    )

    # Step 4: Blend macro-posicional
    def _blend(row: pd.Series) -> float:
        pos = row.get("position_group", "MID")
        w = MACRO_WEIGHTS.get(pos, MACRO_WEIGHTS["MID"])
        of = row.get("obv_for_pt")
        ag = row.get("obv_against_pt")
        if pd.isna(of) and pd.isna(ag):
            return np.nan
        of = of if not pd.isna(of) else 50.0
        ag = ag if not pd.isna(ag) else 50.0
        return w["w_for"] * of + w["w_against"] * ag

    combined["score_OBV"] = combined.apply(_blend, axis=1)
    return combined[["player_id", "player_name", "season_name", "position_group",
                      "obv_for_pt", "obv_against_pt", "score_OBV"]]


# ── Componente PSR ─────────────────────────────────────────────────────────────

def compute_score_psr(
    psr_agg: pd.DataFrame,
    positions: pd.DataFrame,
    percentile_method: str = "average",
    psr_ev_agg: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Implementa §5.2: score_PSR consistente con score_OBV.

    Método (consistente con compute_score_obv):
      1. Si psr_ev_agg disponible (p02_psr_by_event.parquet):
         - Percentiliza PSR dentro de (season, position_group, event_type)
         - Promedia percentiles por (player, season) → score_PSR
      2. Si no disponible (fallback): percentiliza PSR global por (season, position_group)

    La consistencia metodológica evita que la mayor varianza del PSR global
    (std≈29 vs OBV std≈14) produzca artificialmente alpha → 1.0 en la batería P05.

    Input:
        psr_agg:    p02_psr_player_season.parquet (PSR global, fallback)
        positions:  p01_canonical_positions.parquet
        psr_ev_agg: p02_psr_by_event.parquet (PSR por event_type, preferido)
    Output:
        DataFrame con player_id, season_name, score_PSR, has_psr
        score_PSR ∈ [0,100]
    """
    if psr_ev_agg is not None and len(psr_ev_agg) > 0:
        # Método consistente: percentil por event_type → promedio
        df = psr_ev_agg.merge(
            positions[["player_id", "position_group"]],
            on="player_id", how="left"
        )

        df["pct_psr"] = (
            df.groupby(["season_name", "position_group", "event_type"])["psr"]
            .rank(method=percentile_method, pct=True, na_option="keep")
            * 100
        )

        score = (
            df.groupby(["player_id", "season_name"])["pct_psr"]
            .mean()
            .reset_index()
            .rename(columns={"pct_psr": "score_PSR"})
        )
        score["has_psr"] = 1
        print(f"  PSR (consistente, percentil×event_type): {len(score)} player-seasons")
    else:
        # Fallback: percentil global (método anterior)
        df = psr_agg.merge(
            positions[["player_id", "position_group"]],
            on="player_id", how="left"
        )
        df["score_PSR"] = (
            df.groupby(["season_name", "position_group"])["psr"]
            .rank(method=percentile_method, pct=True, na_option="keep")
            * 100
        )
        df["has_psr"] = df["score_PSR"].notna().astype(int)
        score = df[["player_id", "season_name", "score_PSR", "has_psr"]].drop_duplicates(
            subset=["player_id", "season_name"]
        )
        print(f"  PSR (fallback global): {score['has_psr'].sum()} player-seasons")

    return score[["player_id", "season_name", "score_PSR", "has_psr"]].drop_duplicates(
        subset=["player_id", "season_name"]
    )


# ── Función genérica de score_pt ──────────────────────────────────────────────

def compute_score(
    obv_agg: pd.DataFrame,
    psr_agg: pd.DataFrame,
    positions: pd.DataFrame,
    config: ScoreConfig,
    psr_ev_agg: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Función genérica que computa score_pt dado una configuración.
    Llamada desde P05_model_battery.py con distintos alpha y min_minutes.

    Parameters:
        obv_agg:    p02_obv_player_season.parquet
        psr_agg:    p02_psr_player_season.parquet (fallback PSR global)
        positions:  p01_canonical_positions.parquet
        config:     ScoreConfig (alpha, min_minutes, psr_mode, percentile_method)
        psr_ev_agg: p02_psr_by_event.parquet (PSR por event_type, método consistente)

    Returns: DataFrame con columnas
        player_id, player_name, season_name, position_group,
        score_OBV, score_PSR, score_pt, has_psr, obv_for_pt, obv_against_pt
    """
    score_obv_df = compute_score_obv(obv_agg, positions, config.percentile_method)
    score_psr_df = compute_score_psr(
        psr_agg, positions, config.percentile_method, psr_ev_agg=psr_ev_agg
    )

    merged = score_obv_df.merge(
        score_psr_df, on=["player_id", "season_name"], how="left"
    )
    merged["has_psr"] = merged["has_psr"].fillna(0).astype(int)

    # score_pt = (1-α)·score_OBV + α·score_PSR
    # Para jugadores sin PSR: score_pt = score_OBV (α efectivo = 0)
    alpha = config.alpha
    merged["score_pt"] = np.where(
        merged["has_psr"] == 1,
        (1 - alpha) * merged["score_OBV"] + alpha * merged["score_PSR"],
        merged["score_OBV"],
    )
    merged["score_pt"] = merged["score_pt"].clip(0, 100)

    return merged


# ── Validación de output ───────────────────────────────────────────────────────

def validate_score_output(df: pd.DataFrame) -> None:
    """Tests de regresión aplicados al output de componentes (score_OBV, score_PSR)."""
    assert "score_pt" not in df.columns, \
        "[ERROR] score_pt no debería estar en el output de P03 — es decisión de P05"

    valid_obv = df.dropna(subset=["score_OBV"])
    assert (valid_obv["score_OBV"] >= 0).all() and (valid_obv["score_OBV"] <= 100).all(), \
        "score_OBV fuera de [0, 100]"

    cov_360 = df["has_psr"].mean()
    assert cov_360 > 0.20, \
        f"Cobertura PSR = {cov_360:.1%} — menos del 20% de jugadores tienen datos 360"
    print(f"  Cobertura PSR (has_psr=1): {cov_360:.1%}")

    valid_psr = df[df["has_psr"] == 1].dropna(subset=["score_PSR"])
    if len(valid_psr) > 0:
        assert (valid_psr["score_PSR"] >= 0).all() and (valid_psr["score_PSR"] <= 100).all(), \
            "score_PSR fuera de [0, 100]"

    # Distribución no degenerada por season × position
    for (season, pos), grp in df.groupby(["season_name", "position_group"]):
        if len(grp) < 5:
            continue
        std_val = grp["score_OBV"].std()
        assert std_val > 3, \
            f"score_OBV con muy baja varianza en {season}/{pos}: std={std_val:.2f}"

    # Correlación baja entre OBV y PSR (paper §5: ρ ≈ 0.044)
    both = df.dropna(subset=["score_OBV", "score_PSR"])
    if len(both) > 50:
        rho, _ = spearmanr(both["score_OBV"], both["score_PSR"])
        print(f"  ρ(score_OBV, score_PSR) = {rho:.3f}  (paper: ~0.044)")
        if abs(rho) > 0.40:
            print(f"  [AVISO] Correlación alta ({rho:.3f}) entre OBV y PSR — revisar inputs")

    print(f"  [TESTS OK] {len(valid_obv)} filas con score_OBV, {len(valid_psr)} con score_PSR")


# ── Resumen EDA ───────────────────────────────────────────────────────────────

def generate_p03_eda_summary(scores: pd.DataFrame, obv_agg: pd.DataFrame) -> None:
    """Genera p03_eda_summary.csv y p03_eda_event_coverage.csv con estadísticas del diseño."""
    rows = []
    for pos in ["GK", "DEF", "MID", "FWD", "ALL"]:
        sub = scores if pos == "ALL" else scores[scores["position_group"] == pos]
        for col in ["score_OBV", "score_PSR", "obv_for_pt", "obv_against_pt"]:
            s = sub[col].dropna()
            if len(s) == 0:
                continue
            rows.append({
                "position":  pos,
                "metric":    col,
                "n":         len(s),
                "mean":      round(s.mean(), 2),
                "std":       round(s.std(), 2),
                "min":       round(s.min(), 2),
                "p25":       round(s.quantile(0.25), 2),
                "median":    round(s.median(), 2),
                "p75":       round(s.quantile(0.75), 2),
                "max":       round(s.max(), 2),
            })

    summary_df = pd.DataFrame(rows)
    summary_path = OUT_DIR / "p03_eda_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # Coverage por evento en el diseño posición-específico
    all_events = set()
    for d in [AR1_WEIGHTS_FOR, AR1_WEIGHTS_AGAINST]:
        for ev_dict in d.values():
            all_events.update(ev_dict.keys())

    cov_rows = []
    for et in sorted(all_events):
        for pos in ["GK", "DEF", "MID", "FWD"]:
            n = len(obv_agg[(obv_agg["event_type"] == et) & (obv_agg["position_group"] == pos)])
            in_for     = et in AR1_WEIGHTS_FOR.get(pos, {})
            in_against = et in AR1_WEIGHTS_AGAINST.get(pos, {})
            w_for      = AR1_WEIGHTS_FOR.get(pos, {}).get(et, 0.0)
            w_against  = AR1_WEIGHTS_AGAINST.get(pos, {}).get(et, 0.0)
            cov_rows.append({
                "event_type": et, "position": pos, "n_player_seasons": n,
                "in_FOR": in_for, "in_AGAINST": in_against,
                "ar1_for": w_for, "ar1_against": w_against,
                "shot_shrinkage": (et == "Shot" and pos in ("FWD", "MID")),
            })
    cov_df = pd.DataFrame(cov_rows)
    cov_path = OUT_DIR / "p03_eda_event_coverage.csv"
    cov_df.to_csv(cov_path, index=False)

    print(f"\n  === RESUMEN EDA P03 ===")
    print(f"  Diseño aplicado (AR1-proporcional, umbral ≥0.10, sign fix AGAINST):")
    for pos in ["GK", "DEF", "MID", "FWD"]:
        for_events = list(AR1_WEIGHTS_FOR.get(pos, {}).keys())
        ag_events  = list(AR1_WEIGHTS_AGAINST.get(pos, {}).keys())
        print(f"    {pos} FOR ({len(for_events)}): {for_events}")
        print(f"    {pos} AGAINST ({len(ag_events)}): {ag_events}")
    obv_all = scores["score_OBV"].dropna()
    print(f"  Distribución score_OBV (ALL):")
    print(f"    mean={obv_all.mean():.2f}  std={obv_all.std():.2f}  "
          f"min={obv_all.min():.2f}  max={obv_all.max():.2f}")
    print(f"  Outputs EDA: {summary_path.name}, {cov_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> pd.DataFrame:
    print("=" * 60)
    print("P03 — Score Component Computation (score_OBV + score_PSR)")
    print("=" * 60)
    print("  NOTA: este módulo NO computa score_pt. La ponderación alpha")
    print("  se decide en P05_model_battery.py tras evaluar la batería de modelos.")

    # Verificar inputs
    for path, name in [(OBV_AGG_PATH, "P02 OBV output"), (PSR_AGG_PATH, "P02 PSR output"),
                       (POS_PATH, "P01 canonical positions")]:
        assert path.exists(), \
            f"[ERROR] Input no encontrado: {path} ({name})\nCorrer P02/P01 primero."

    obv_agg   = pd.read_parquet(OBV_AGG_PATH)
    psr_agg   = pd.read_parquet(PSR_AGG_PATH)
    positions = pd.read_parquet(POS_PATH)

    psr_ev_agg = None
    if PSR_EV_AGG_PATH.exists():
        psr_ev_agg = pd.read_parquet(PSR_EV_AGG_PATH)
        print(f"  PSR by event_type: {len(psr_ev_agg)} filas — usando método consistente")
    else:
        print(f"  [AVISO] {PSR_EV_AGG_PATH} no encontrado — usando fallback PSR global")

    print(f"\n  OBV agg: {len(obv_agg):,} filas, {obv_agg['player_id'].nunique()} jugadores")
    print(f"  PSR agg: {psr_agg['psr'].notna().sum()} player-seasons con PSR válido")
    print(f"  Posiciones canónicas: {len(positions)} jugadores")

    # Computar componentes independientemente (sin alpha)
    score_obv_df = compute_score_obv(obv_agg, positions)
    score_psr_df = compute_score_psr(psr_agg, positions, psr_ev_agg=psr_ev_agg)

    scores = score_obv_df.merge(
        score_psr_df, on=["player_id", "season_name"], how="left"
    )
    scores["has_psr"] = scores["has_psr"].fillna(0).astype(int)

    validate_score_output(scores)
    generate_p03_eda_summary(scores, obv_agg)

    # Guardar solo componentes — sin score_pt
    out_cols = [
        "player_id", "player_name", "season_name", "position_group",
        "obv_for_pt", "obv_against_pt", "score_OBV", "score_PSR", "has_psr",
    ]
    out_path = OUT_DIR / "p03_score_player_season.parquet"
    scores[out_cols].to_parquet(out_path, index=False)

    print(f"\n  Output: {out_path}")
    print(f"  Columnas: {out_cols}")
    print(f"  Filas: {len(scores)}")
    print(f"\n  Para seleccionar alpha y entrenar modelos, ejecutar:")
    print(f"    python P05_model_battery.py")

    return scores[out_cols]


if __name__ == "__main__":
    main()

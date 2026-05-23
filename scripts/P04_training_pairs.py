"""
P04 — Training Pairs (Set D)
==============================
Objetivo:
    Construir el dataset de entrenamiento para el modelo de proyección.
    Set D = jugadores observados en AMBAS temporadas 22/23 y 23/24, con
    al menos MIN_MINUTES minutos en cada una.

    IMPORTANTE — Separación de responsabilidades:
      - P04 produce pares de componentes (score_OBV, score_PSR) para cada
        jugador en t=0 y t=1. NO computa score_pt.
      - El umbral MIN_MINUTES se evalúa en 4 variantes uniformes (250,500,750,1000)
        MÁS 2 variantes con umbrales por posición. P05 decide cuál usar.
      - La ponderación alpha también la decide P05. P04 no asume ningún alpha.

    Umbrales por posición: GK usa 250 min (titulares EPL superan siempre ese
    umbral; backups con <250 min tienen score_OBV poco fiable). DEF/MID/FWD
    usan el umbral estándar. Se generan dos variantes:
      - pos_GK250_OFP750:  GK≥250, DEF/MID/FWD≥750
      - pos_GK250_OFP1000: GK≥250, DEF/MID/FWD≥1000

    FIX: get_minutes() agrega minutos por player_id con groupby+sum para
    manejar correctamente tanto transfers a mitad de temporada (suma legítima)
    como filas duplicadas en PSF (idempotente tras drop_duplicates).

Outputs:
    outputs/pipeline/p04_training_pairs_{min_min}min.parquet   (4 archivos)
    outputs/pipeline/p04_training_pairs_pos_GK250_OFP750.parquet
    outputs/pipeline/p04_training_pairs_pos_GK250_OFP1000.parquet
    Columnas: player_id, player_name, position_group, age_t0,
              score_OBV_t0, score_PSR_t0, has_psr_t0,
              score_OBV_t1, score_PSR_t1, has_psr_t1,
              min_t0, min_t1, set_label

Upstream:
    P03: outputs/pipeline/p03_score_player_season.parquet
    STATIC: data/processed/statsbomb_legacy/df_final_market_value.csv  (para age)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
SCORES_PATH = REPO / "outputs/pipeline/p03_score_player_season.parquet"
MV_PATH     = REPO / "data/processed/statsbomb_legacy/df_final_market_value.csv"
PSF_PATH    = REPO / "data/processed/modeling/player_season_features.parquet"
OUT_DIR     = REPO / "outputs/pipeline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEASON_T0 = "2022/2023"
SEASON_T1 = "2023/2024"

# Umbrales uniformes — P04 genera un parquet por cada uno
MIN_MINUTES_OPTIONS = [250, 500, 750, 1000]

# Umbrales por posición — GK tiene umbral propio más bajo
POS_THRESHOLD_CONFIGS: list[dict[str, int]] = [
    {"GK": 250, "DEF": 750,  "MID": 750,  "FWD": 750,  "_label": "pos_GK250_OFP750"},
    {"GK": 250, "DEF": 1000, "MID": 1000, "FWD": 1000, "_label": "pos_GK250_OFP1000"},
]


def get_minutes(season_name: str) -> pd.DataFrame:
    """
    Retorna DataFrame con (player_id, minutes_played) para una temporada.
    Fuente primaria: player_season_features.parquet (PSF).

    FIX: elimina filas exactamente duplicadas y suma minutos por player_id
    para manejar mid-season transfers y duplicados en PSF.
    """
    assert PSF_PATH.exists(), \
        f"[ERROR] {PSF_PATH} no encontrado — necesario para minutos jugados."
    psf = pd.read_parquet(PSF_PATH)
    mask = psf["season_name"].str.replace("/", "/", regex=False).str.strip() == season_name
    sub = psf[mask & (psf["competition_name"] == "Premier League")].dropna(subset=["minutes_played"])
    # drop_duplicates elimina filas idénticas (bug PSF); groupby sum acumula
    # minutos cuando el mismo player_id aparece en varios clubes en la misma temporada.
    sub = sub.drop_duplicates()
    return (
        sub.groupby("player_id", as_index=False)["minutes_played"]
           .sum()
    )


def get_age_at_season(mv_path: Path, season_name: str) -> dict[int, float]:
    """Retorna lookup player_id → age para una temporada desde df_final_market_value."""
    mv = pd.read_csv(mv_path)
    mv_s = mv[mv["season_name"] == season_name].drop_duplicates("player_id")
    return dict(zip(mv_s["player_id"].astype(int), mv_s["age"].astype(float)))


def build_training_pairs(min_minutes: int) -> pd.DataFrame:
    """
    Construye Set D: pares de componentes (score_OBV, score_PSR) para t=0 y t=1.

    NO computa score_pt. P05 lo computará para cada (alpha, min_minutes) evaluado.

    Parameters:
        min_minutes: umbral mínimo de minutos en AMBAS temporadas.

    Returns:
        DataFrame con columnas:
          player_id, player_name, position_group, age_t0,
          score_OBV_t0, score_PSR_t0, has_psr_t0,
          score_OBV_t1, score_PSR_t1, has_psr_t1,
          min_t0, min_t1, set_label
    """
    assert SCORES_PATH.exists(), \
        f"[ERROR] Scores no encontrados: {SCORES_PATH}\nCorrer P03 primero."

    scores = pd.read_parquet(SCORES_PATH)

    assert "score_pt" not in scores.columns, \
        "[ERROR] score_pt encontrado en p03_score_player_season.parquet — " \
        "re-ejecutar P03 (no debe computar score_pt)."

    # Separar temporadas — usar componentes, no score_pt
    s0 = scores[scores["season_name"] == SEASON_T0][
        ["player_id", "player_name", "position_group",
         "score_OBV", "score_PSR", "has_psr"]
    ].rename(columns={
        "score_OBV": "score_OBV_t0",
        "score_PSR": "score_PSR_t0",
        "has_psr":   "has_psr_t0",
    })

    s1 = scores[scores["season_name"] == SEASON_T1][
        ["player_id", "score_OBV", "score_PSR", "has_psr"]
    ].rename(columns={
        "score_OBV": "score_OBV_t1",
        "score_PSR": "score_PSR_t1",
        "has_psr":   "has_psr_t1",
    })

    # Obtener minutos
    min_t0 = get_minutes(SEASON_T0).rename(columns={"minutes_played": "min_t0"})
    min_t1 = get_minutes(SEASON_T1).rename(columns={"minutes_played": "min_t1"})

    # Construir pares
    pairs = (
        s0.merge(s1,     on="player_id", how="inner")
          .merge(min_t0, on="player_id", how="left")
          .merge(min_t1, on="player_id", how="left")
    )

    # Aplicar filtro de minutos
    pairs = pairs[
        (pairs["min_t0"] >= min_minutes) &
        (pairs["min_t1"] >= min_minutes)
    ].copy()

    # Añadir age_t0
    age_lookup = get_age_at_season(MV_PATH, SEASON_T0)
    pairs["age_t0"] = pairs["player_id"].map(age_lookup)

    # Fallback de age desde PSF si falta
    missing_age = pairs["age_t0"].isna().sum()
    if missing_age > 0:
        psf = pd.read_parquet(PSF_PATH)
        psf_age = (
            psf[(psf["season_name"] == SEASON_T0) & (psf["competition_name"] == "Premier League")]
            .drop_duplicates("player_id")
            .set_index("player_id")["age"]
        )
        pairs.loc[pairs["age_t0"].isna(), "age_t0"] = (
            pairs.loc[pairs["age_t0"].isna(), "player_id"].map(psf_age)
        )
        print(f"  [INFO] {missing_age} edades completadas desde PSF")

    pairs["has_psr_t0"] = pairs["has_psr_t0"].fillna(0).astype(int)
    pairs["has_psr_t1"] = pairs["has_psr_t1"].fillna(0).astype(int)
    pairs["set_label"] = "D"
    pairs = pairs.dropna(subset=["score_OBV_t0", "score_OBV_t1"])
    pairs = pairs.drop_duplicates("player_id")

    return pairs.reset_index(drop=True)


def build_training_pairs_by_position(min_by_pos: dict[str, int]) -> pd.DataFrame:
    """
    Construye pares con umbrales de minutos diferenciados por posición.

    Parameters:
        min_by_pos: {position_group: min_minutes}, e.g.
                    {"GK": 250, "DEF": 1000, "MID": 1000, "FWD": 1000}

    Estrategia: construir base con el umbral mínimo, luego aplicar filtro
    posicional estricto sobre min_t0 y min_t1 para cada grupo.
    """
    min_floor = min(v for k, v in min_by_pos.items() if not k.startswith("_"))
    base = build_training_pairs(min_floor)
    parts = []
    for pos, mm in min_by_pos.items():
        if pos.startswith("_"):
            continue
        part = base[
            (base["position_group"] == pos) &
            (base["min_t0"] >= mm) &
            (base["min_t1"] >= mm)
        ]
        parts.append(part)
    return pd.concat(parts, ignore_index=True).drop_duplicates("player_id").reset_index(drop=True)


def validate_pairs(pairs: pd.DataFrame, label: str) -> None:
    """Tests de regresión del Set D (componentes, sin score_pt)."""
    assert "score_pt" not in pairs.columns, \
        "[ERROR] score_pt no debería estar en pares de P04"
    assert "score_t0" not in pairs.columns and "score_t1" not in pairs.columns, \
        "[ERROR] Columnas score_t0/score_t1 reemplazadas por score_OBV/PSR por temporada"

    N = len(pairs)
    assert 30 <= N <= 600, \
        f"N_pairs={N} ({label}) — fuera del rango esperado [30, 600]"
    assert pairs["score_OBV_t0"].between(0, 100).all(), "score_OBV_t0 fuera de [0, 100]"
    assert pairs["score_OBV_t1"].between(0, 100).all(), "score_OBV_t1 fuera de [0, 100]"
    assert (pairs["age_t0"].between(15, 45)).all(), \
        f"age_t0 fuera de rango: {pairs[~pairs['age_t0'].between(15, 45)][['player_id','age_t0']]}"
    assert pairs["player_id"].duplicated().sum() == 0, "player_id duplicado en Set D"
    assert pairs["position_group"].isin({"GK", "DEF", "MID", "FWD"}).all(), \
        "position_group con valores inesperados"
    for pos in ["GK", "DEF", "MID", "FWD"]:
        n_pos = (pairs["position_group"] == pos).sum()
        assert n_pos >= 3, \
            f"Solo {n_pos} jugadores de posición {pos} — insuficiente para efectos fijos"

    from scipy.stats import spearmanr
    rho, _ = spearmanr(pairs["score_OBV_t0"], pairs["score_OBV_t1"])
    print(f"  ρ(score_OBV_t0, score_OBV_t1) = {rho:.3f}")
    psr_both = pairs[(pairs["has_psr_t0"] == 1) & (pairs["has_psr_t1"] == 1)]
    if len(psr_both) > 10:
        rho_psr, _ = spearmanr(psr_both["score_PSR_t0"], psr_both["score_PSR_t1"])
        print(f"  ρ(score_PSR_t0, score_PSR_t1) = {rho_psr:.3f}  (N={len(psr_both)})")

    psr_cov = pairs["has_psr_t0"].mean()
    print(f"  Cobertura PSR en t0: {psr_cov:.1%}")

    print(f"  [TESTS OK] N={N} pares, label={label}")
    print(f"  Distribución posicional:")
    for pos in ["GK", "DEF", "MID", "FWD"]:
        n = (pairs["position_group"] == pos).sum()
        print(f"    {pos}: {n} ({n/len(pairs):.1%})")


def main() -> dict[str, pd.DataFrame]:
    """Genera pares para todos los umbrales de minutos. P05 decide cuál usar."""
    print("=" * 60)
    print("P04 — Training Pairs (Set D) — todos los umbrales")
    print("=" * 60)
    print(f"  Umbrales uniformes: {MIN_MINUTES_OPTIONS}")
    print(f"  Configs por posición: {[c['_label'] for c in POS_THRESHOLD_CONFIGS]}")
    print("  FIX: get_minutes() con drop_duplicates+groupby sum (bug PSF corregido)")
    print("  NOTA: P04 genera componentes (score_OBV, score_PSR). Alpha se decide en P05.\n")

    results: dict[str, pd.DataFrame] = {}

    # Umbrales uniformes
    for mm in MIN_MINUTES_OPTIONS:
        pairs = build_training_pairs(mm)
        label = f"{mm}min"
        validate_pairs(pairs, label)
        out_path = OUT_DIR / f"p04_training_pairs_{label}.parquet"
        pairs.to_parquet(out_path, index=False)
        print(f"  Output: {out_path}  (N={len(pairs)})\n")
        results[label] = pairs

    # Configs por posición
    for cfg in POS_THRESHOLD_CONFIGS:
        lbl = cfg["_label"]
        pairs = build_training_pairs_by_position(cfg)
        validate_pairs(pairs, lbl)
        out_path = OUT_DIR / f"p04_training_pairs_{lbl}.parquet"
        pairs.to_parquet(out_path, index=False)
        print(f"  Output: {out_path}  (N={len(pairs)})\n")
        results[lbl] = pairs

    print("=" * 60)
    print(f"  Resumen N pares:")
    for lbl, df in results.items():
        gk = (df["position_group"]=="GK").sum()
        print(f"    {lbl:30s} → N={len(df)}  (GK={gk})")

    # ── Resumen EDA P04 ─────────────────────────────────────────────────────
    from scipy.stats import spearmanr as _sp
    eda_rows = []
    for lbl, df in results.items():
        rho_obv, _  = _sp(df["score_OBV_t0"], df["score_OBV_t1"])
        psr_b = df[(df["has_psr_t0"] == 1) & (df["has_psr_t1"] == 1)]
        rho_psr = _sp(psr_b["score_PSR_t0"], psr_b["score_PSR_t1"])[0] if len(psr_b) > 10 else float("nan")
        eda_rows.append({
            "label":        lbl,
            "n_pairs":      len(df),
            "ar1_score_OBV": round(float(rho_obv), 3),
            "ar1_score_PSR": round(float(rho_psr), 3),
            "psr_coverage_t0": round(df["has_psr_t0"].mean(), 3),
            "age_t0_mean":  round(df["age_t0"].mean(), 1) if "age_t0" in df.columns else float("nan"),
            "n_pos_GK":  (df["position_group"] == "GK").sum(),
            "n_pos_DEF": (df["position_group"] == "DEF").sum(),
            "n_pos_MID": (df["position_group"] == "MID").sum(),
            "n_pos_FWD": (df["position_group"] == "FWD").sum(),
        })
    eda_df = pd.DataFrame(eda_rows)
    eda_path = OUT_DIR / "p04_eda_summary.csv"
    eda_df.to_csv(eda_path, index=False)

    print(f"\n  === RESUMEN EDA P04 ===")
    print(eda_df[["label", "n_pairs", "ar1_score_OBV", "ar1_score_PSR", "n_pos_GK"]].to_string(index=False))
    print(f"  Output EDA: {eda_path}")

    return results


if __name__ == "__main__":
    main()

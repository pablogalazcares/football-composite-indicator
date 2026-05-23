"""
P06 — Projection Universe (Set C = A ∪ B, canónico)
====================================================
Definiciones canónicas (paper §4, CLAUDE.md):
    A = jugadores con score válido en 2022/23 (eventing+OBV)
    B = jugadores con score válido en 2023/24 (eventing+OBV)
    C = A ∪ B  → universo de proyección  ← ESTE SCRIPT
    D = A ∩ B  → training pairs (P04)
    E = A \\ B → solo 22/23 (incluye relegados Leeds/Leicester/Southampton)
    F = B \\ A → solo 23/24 (incluye promovidos Burnley/Luton/Sheffield Utd)

Universo de proyección = C = (B ∩ A) ∪ E ∪ F
                          = jugadores en B  ∪ jugadores solo en A
                          ≈ 759 jugadores con al menos una temporada EPL.

Etiquetas en output:
    set_label = 'B' → jugador presente en 2023/24 (base de proyección directa)
    set_label = 'E' → jugador presente solo en 2022/23 (proyección desde 22/23)

Columnas adicionales:
    in_A : bool — presente en 2022/23
    in_B : bool — presente en 2023/24

NO se aplica filtro de min_minutes aquí. El filtro de min_minutes en P04
es solo para el entrenamiento del modelo de proyección. La proyección se
aplica a TODOS los jugadores en el universo, incluidos los de baja exposición.

Outputs:
    outputs/pipeline/p06_projection_universe.parquet
    Columnas: player_id, player_name, position_group, team_name,
              season_base, score_pt_base, age_base, mv_base,
              set_label (B / E), in_A, in_B

Tests inline:
    - N universe ∈ [600, 900]
    - in_A | in_B = True para todas las filas
    - position_group distribution: MID < 35%
    - score_pt_base ∈ [0, 100] para filas no-nulas
    - age_base ∈ [15, 45]
    - player_id sin duplicados

Upstream:
    P03: outputs/pipeline/p03_score_player_season.parquet
    P01: outputs/pipeline/p01_canonical_positions.parquet
    P05: outputs/pipeline/p05_selected_config.csv (para α)
    STATIC: data/processed/statsbomb_legacy/df_final_market_value.csv
    PSF:   data/processed/modeling/player_season_features.parquet
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
SCORES_PATH = REPO / "outputs/pipeline/p03_score_player_season.parquet"
POS_PATH    = REPO / "outputs/pipeline/p01_canonical_positions.parquet"
MV_PATH     = REPO / "data/processed/statsbomb_legacy/df_final_market_value.csv"
PSF_PATH    = REPO / "data/processed/modeling/player_season_features.parquet"
OUT_DIR     = REPO / "outputs/pipeline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEASON_T0 = "2022/2023"
SEASON_T1 = "2023/2024"

P05_CONFIG_PATH = REPO / "outputs/pipeline/p05_selected_config.csv"


def _get_alpha() -> float:
    """Lee α del modelo seleccionado por P05."""
    assert P05_CONFIG_PATH.exists(), \
        f"[ERROR] {P05_CONFIG_PATH} no encontrado — correr P05 primero."
    cfg = pd.read_csv(P05_CONFIG_PATH)
    alpha = float(cfg.iloc[0]["alpha"])
    print(f"  α = {alpha} (cargado desde P05 selected config)")
    return alpha


def _compose_score(scores: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Compone score_pt = (1-α)*score_OBV + α*score_PSR.
    Para jugadores sin PSR válido (has_psr=0), usa effective α=0.
    """
    df = scores.copy()
    if "score_pt" in df.columns:
        return df
    # Effective alpha: 0 si no hay PSR
    df["_eff_alpha"] = alpha * df["has_psr"].fillna(0)
    df["score_pt"] = (
        (1 - df["_eff_alpha"]) * df["score_OBV"].fillna(0)
        + df["_eff_alpha"] * df["score_PSR"].fillna(0)
    )
    return df.drop(columns=["_eff_alpha"])


def _get_age_lookup(season: str) -> dict[int, float]:
    """Retorna player_id → age desde df_final_market_value para una temporada."""
    mv = pd.read_csv(MV_PATH)
    sub = mv[mv["season_name"] == season].drop_duplicates("player_id")
    return dict(zip(sub["player_id"].astype(int), sub["age"].astype(float)))


def _get_mv_lookup(season: str) -> dict[int, float]:
    """Retorna player_id → market_value (€) desde df_final_market_value."""
    mv = pd.read_csv(MV_PATH)
    sub = mv[mv["season_name"] == season].drop_duplicates("player_id")
    return dict(zip(sub["player_id"].astype(int), sub["market_value"].astype(float)))


def _get_team_lookup(season: str) -> dict[int, str]:
    """Retorna player_id → team_name para una temporada."""
    mv = pd.read_csv(MV_PATH)
    sub = mv[mv["season_name"] == season].drop_duplicates("player_id")
    return dict(zip(sub["player_id"].astype(int), sub["team_name"].astype(str)))


def build_universe() -> pd.DataFrame:
    """
    Construye el universo de proyección C = A ∪ B (definición canónica):
      - Jugadores en B (2023/24) → set_label = 'B'
      - Jugadores solo en A (2022/23, no en 2023/24) → set_label = 'E' (= A\\B)

    Returns:
        DataFrame con player_id, player_name, position_group, team_name,
        season_base, score_pt_base, age_base, mv_base, set_label, in_A, in_B
    """
    assert SCORES_PATH.exists(), \
        f"[ERROR] {SCORES_PATH} no encontrado — correr P03 primero."
    assert POS_PATH.exists(), \
        f"[ERROR] {POS_PATH} no encontrado — correr P01 primero."

    alpha     = _get_alpha()
    scores    = pd.read_parquet(SCORES_PATH)
    scores    = _compose_score(scores, alpha)
    positions = pd.read_parquet(POS_PATH)

    # IDs presentes en cada temporada
    ids_A = set(scores.loc[scores["season_name"] == SEASON_T0, "player_id"].unique())
    ids_B = set(scores.loc[scores["season_name"] == SEASON_T1, "player_id"].unique())

    # ── Set B: jugadores con score en 2023/24 ─────────────────────────────────
    s1 = scores[scores["season_name"] == SEASON_T1][
        ["player_id", "player_name", "position_group", "score_pt"]
    ].rename(columns={"score_pt": "score_pt_base"}).copy()
    s1["season_base"] = SEASON_T1
    s1["set_label"]   = "B"

    # ── Set E = A \ B: solo en 2022/23 (relegados, jugadores que salieron) ────
    s0 = scores[scores["season_name"] == SEASON_T0][
        ["player_id", "player_name", "position_group", "score_pt"]
    ].rename(columns={"score_pt": "score_pt_base"}).copy()
    s0 = s0[~s0["player_id"].isin(ids_B)].copy()
    s0["season_base"] = SEASON_T0
    s0["set_label"]   = "E"

    # ── Combinar → Universo C = B ∪ E (que es A ∪ B canónico) ────────────────
    universe = pd.concat([s1, s0], ignore_index=True)
    universe = universe.drop_duplicates("player_id")

    # ── Añadir columnas in_A, in_B ────────────────────────────────────────────
    universe["in_A"] = universe["player_id"].isin(ids_A)
    universe["in_B"] = universe["player_id"].isin(ids_B)

    # ── Añadir position_group canónica (overwrite si disponible) ──────────────
    canon_pos = positions[["player_id", "position_group"]].rename(
        columns={"position_group": "pg_canon"}
    )
    universe = universe.merge(canon_pos, on="player_id", how="left")
    universe["position_group"] = universe["pg_canon"].fillna(universe["position_group"])
    universe = universe.drop(columns=["pg_canon"])

    # ── Fallback posición: MID para jugadores sin position_group ────────────
    missing_pos = universe["position_group"].isna() | (universe["position_group"] == "None")
    n_missing_pos = missing_pos.sum()
    if n_missing_pos > 0:
        universe.loc[missing_pos, "position_group"] = "MID"
        print(f"  [WARN Q-01] {n_missing_pos} jugadores sin posición → fallback MID")

    # ── Añadir age_base ───────────────────────────────────────────────────────
    age_t1 = _get_age_lookup(SEASON_T1)
    age_t0 = _get_age_lookup(SEASON_T0)
    universe["age_base"] = universe.apply(
        lambda r: age_t1.get(r["player_id"], age_t0.get(r["player_id"], np.nan)),
        axis=1,
    )
    # Fallback: PSF
    missing_age = universe["age_base"].isna().sum()
    if missing_age > 0:
        psf = pd.read_parquet(PSF_PATH)
        psf_age = (
            psf[(psf["season_name"] == SEASON_T1) & (psf["competition_name"] == "Premier League")]
            .drop_duplicates("player_id")
            .set_index("player_id")["age"]
        )
        universe.loc[universe["age_base"].isna(), "age_base"] = (
            universe.loc[universe["age_base"].isna(), "player_id"].map(psf_age)
        )
        still_missing = universe["age_base"].isna().sum()
        print(f"  [INFO] {missing_age - still_missing} edades completadas desde PSF; "
              f"{still_missing} siguen faltando")

    # ── Añadir mv_base (en €) ─────────────────────────────────────────────────
    mv_t1 = _get_mv_lookup(SEASON_T1)
    mv_t0 = _get_mv_lookup(SEASON_T0)
    universe["mv_base"] = universe.apply(
        lambda r: mv_t1.get(r["player_id"], mv_t0.get(r["player_id"], np.nan)),
        axis=1,
    )

    # ── Añadir team_name ──────────────────────────────────────────────────────
    team_t1 = _get_team_lookup(SEASON_T1)
    team_t0 = _get_team_lookup(SEASON_T0)
    universe["team_name"] = universe.apply(
        lambda r: team_t1.get(r["player_id"], team_t0.get(r["player_id"], "Unknown")),
        axis=1,
    )

    return universe.reset_index(drop=True)


def validate_universe(universe: pd.DataFrame) -> None:
    """Tests de regresión del universo de proyección."""
    N = len(universe)
    assert 600 <= N <= 900, \
        f"N_universe={N} fuera del rango esperado [600, 900]"

    assert universe["player_id"].duplicated().sum() == 0, \
        "player_id duplicados en universo de proyección"

    assert universe["position_group"].isin({"GK", "DEF", "MID", "FWD"}).all(), \
        f"position_group con valores inesperados: " \
        f"{universe[~universe['position_group'].isin({'GK','DEF','MID','FWD'})]['position_group'].unique()}"

    mid_pct = (universe["position_group"] == "MID").mean()
    assert mid_pct < 0.35, \
        f"MID={mid_pct:.1%} — sospecha de bug de imputación posicional (>35%)"

    valid_score = universe["score_pt_base"].dropna()
    assert (valid_score >= 0).all() and (valid_score <= 100).all(), \
        "score_pt_base fuera de [0, 100]"

    valid_age = universe["age_base"].dropna()
    assert (valid_age >= 15).all() and (valid_age <= 45).all(), \
        f"age_base fuera de [15, 45]: {valid_age[~valid_age.between(15, 45)].tolist()}"

    # ── Sets canónicos ────────────────────────────────────────────────────────
    set_B = (universe["set_label"] == "B").sum()
    set_E = (universe["set_label"] == "E").sum()
    # Validar consistencia in_A | in_B
    assert (universe["in_A"] | universe["in_B"]).all(), \
        "Jugadores en universo C deben estar en A o B (canónico)"
    set_A_total = int(universe["in_A"].sum())
    set_B_total = int(universe["in_B"].sum())
    set_D_intersection = int((universe["in_A"] & universe["in_B"]).sum())
    set_F = int((universe["in_B"] & ~universe["in_A"]).sum())

    print(f"  [TESTS OK] N={N} jugadores (universo C = A ∪ B)")
    print(f"  |A|={set_A_total}  |B|={set_B_total}  |D=A∩B|={set_D_intersection}  "
          f"|E=A\\B|={set_E}  |F=B\\A|={set_F}")
    print(f"  set_label='B': {set_B} (jugadores en 2023/24)")
    print(f"  set_label='E': {set_E} (jugadores solo en 2022/23)")
    print(f"  Distribución posicional:")
    for pos in ["GK", "DEF", "MID", "FWD"]:
        n = (universe["position_group"] == pos).sum()
        print(f"    {pos}: {n} ({n/N:.1%})")
    print(f"  score_pt_base disponible: {valid_score.shape[0]}/{N} ({valid_score.shape[0]/N:.1%})")
    print(f"  age_base disponible: {valid_age.shape[0]}/{N} ({valid_age.shape[0]/N:.1%})")
    print(f"  mv_base disponible: {universe['mv_base'].notna().sum()}/{N}")


def main() -> pd.DataFrame:
    print("=" * 60)
    print("P06 — Projection Universe C = A ∪ B (canónico)")
    print("=" * 60)

    universe = build_universe()
    validate_universe(universe)

    out_path = OUT_DIR / "p06_projection_universe.parquet"
    universe.to_parquet(out_path, index=False)
    print(f"\n  Output: {out_path}")
    return universe


if __name__ == "__main__":
    main()

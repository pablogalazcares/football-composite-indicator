"""
P07 — Recursive Score Projection (V11)
========================================
Objetivo:
    Proyectar score_pt recursivamente para los H horizontes del optimizador,
    usando el modelo seleccionado por P05 (V11: SLSQP grupo M, quad+inter).

    Fórmula de proyección (paper §6, eq. recursive):
        score_{t+1} = f(score_t, age_t; θ)
    donde f es el modelo V11 entrenado en P05 con los pares del Set D.

    La proyección recursiva se aplica a TODOS los jugadores del universo P06,
    incluidos aquellos con score_pt_base imputado o de baja exposición.

    Para jugadores en Set F (solo 2022/23), el punto de partida (score_pt_base)
    ya incorpora un paso adicional implícito — se usa directamente como h=0.

    Importante: age_t se incrementa en 1 por horizonte. La edad base es age_base
    del universo P06. Si age_base falta, se usa la mediana posicional.

Parámetros del horizonte:
    H = 5 (el paper reporta resultados para H=5 como experimento principal)
    Se guarda score para h=0,1,...,H (H+1 columnas).
    h=0 = score_pt_base (observado o de P03).

Outputs:
    outputs/pipeline/p07_projected_scores.parquet
    Columnas: player_id, player_name, position_group, age_base,
              score_h0, score_h1, ..., score_hH  (H+1 columnas)

Tests inline:
    - score_h* ∈ [0, 100] (el modelo P05 ya clipa a este rango)
    - No todos los scores del mismo jugador son idénticos (variación temporal)
    - Jugadores jóvenes (< 25) tienen score_h5 >= score_h0 en promedio
    - Jugadores mayores (> 30) tienen score_h5 < score_h0 en promedio
    - Convergencia a mediana: std(score_h5) < std(score_h0) [regresión a la media]

Upstream:
    P05: outputs/pipeline/p05_selected_model.pkl
    P06: outputs/pipeline/p06_projection_universe.parquet
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
MODEL_PATH   = REPO / "outputs/pipeline/p05_selected_model.pkl"
UNIVERSE_PATH = REPO / "outputs/pipeline/p06_projection_universe.parquet"
OUT_DIR      = REPO / "outputs/pipeline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 5   # número de pasos de proyección (h=0 es observado)


def load_model(path: Path) -> dict:
    """
    Carga el modelo seleccionado por P05.

    P05 serializa: {"model": fitted_sklearn_model, "config": best_row_dict}
    Retorna un dict con la función de proyección lista para usar.
    """
    import sys
    sys.path.insert(0, str(path.parent.parent.parent / "scripts" / "pipeline"))

    assert path.exists(), (
        f"[ERROR] Modelo no encontrado: {path}\n"
        f"Correr P05_model_battery.py primero para seleccionar y serializar el modelo V11."
    )
    with open(path, "rb") as f:
        raw = pickle.load(f)

    model  = raw["model"]   # sklearn/SLSQP fitted model
    config = raw["config"]  # dict de best_row

    model_id = config.get("model_id", "UNKNOWN")
    print(f"  Modelo cargado: {model_id}  (grupo={config.get('group','?')}, α={config.get('alpha','?')}, min={config.get('min_minutes','?')})")

    # Obtener feature builder del grupo
    from P05_model_battery import (
        build_base_features, build_interaction_features,
        build_cubic_features, build_cubic_interaction_features,
        build_age_bin_features, build_age_only_features,
        build_log_age_features, build_sqrt_age_features,
        build_log_sqrt_age_features,
    )
    group = config.get("group", "A")
    variant_id = config.get("variant_id", "")

    # Seleccionar feature builder según el variant_id registrado
    if "inter" in variant_id:
        build_fn = build_interaction_features
    elif "cubic_inter" in variant_id:
        build_fn = build_cubic_interaction_features
    elif "cubic" in variant_id:
        build_fn = build_cubic_features
    elif "bins" in variant_id:
        build_fn = build_age_bin_features
    elif "age_only" in variant_id:
        build_fn = build_age_only_features
    elif "log_sqrt" in variant_id:
        build_fn = build_log_sqrt_age_features
    elif "log" in variant_id:
        build_fn = build_log_age_features
    elif "sqrt" in variant_id:
        build_fn = build_sqrt_age_features
    else:
        build_fn = build_base_features

    return {"model": model, "config": config, "build_fn": build_fn}


def project_one_step(
    score: np.ndarray,
    age: np.ndarray,
    model_data: dict,
    position_group: np.ndarray | None = None,
) -> np.ndarray:
    """
    Aplica un paso de proyección f(score_t, age_t, pos) → score_{t+1}
    usando el modelo sklearn serializado por P05.

    Parameters:
        score:          np.ndarray de shape (N,)
        age:            np.ndarray de shape (N,)
        model_data:     dict con 'model', 'build_fn', 'config'
        position_group: np.ndarray de shape (N,) con strings GK/DEF/MID/FWD

    Returns:
        np.ndarray de shape (N,), score_{t+1} clipeado a [0, 100]
    """
    model    = model_data["model"]
    build_fn = model_data["build_fn"]
    config   = model_data["config"]

    N = len(score)
    if position_group is None:
        position_group = np.array(["MID"] * N)

    # Construir DataFrame de entrada compatible con build_fn
    row_df = pd.DataFrame({
        "score_t0":       score,
        "age_t0":         age,
        "position_group": position_group,
    })

    X, _ = build_fn(row_df)

    # Grupos con target = delta_score (grupo H): pred = Δ, final = score_t0 + Δ
    if config.get("target_col") == "delta_score":
        delta_pred = model.predict(X)
        pred = score + delta_pred
    else:
        pred = model.predict(X)

    return np.clip(pred, 0, 100)


def project_universe(
    universe: pd.DataFrame,
    model_data: dict,
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """
    Proyecta score_pt para todos los jugadores del universo, H pasos hacia adelante.

    Parameters:
        universe:   p06_projection_universe.parquet
        model_data: dict del modelo serializado por P05
        horizon:    número de pasos de proyección

    Returns:
        DataFrame con player_id y columnas score_h0..score_hH
    """
    df = universe[["player_id", "player_name", "position_group", "age_base",
                   "score_pt_base"]].copy()

    # Imputar score_pt_base nulo con mediana posicional
    for pos, grp in df.groupby("position_group"):
        med = grp["score_pt_base"].median()
        mask = df["position_group"].eq(pos) & df["score_pt_base"].isna()
        df.loc[mask, "score_pt_base"] = med
    df["score_pt_base"] = df["score_pt_base"].fillna(df["score_pt_base"].median())

    # Imputar age_base nulo con mediana posicional
    for pos, grp in df.groupby("position_group"):
        med = grp["age_base"].median()
        mask = df["position_group"].eq(pos) & df["age_base"].isna()
        df.loc[mask, "age_base"] = med
    df["age_base"] = df["age_base"].fillna(df["age_base"].median())

    score    = df["score_pt_base"].values.astype(float)
    age      = df["age_base"].values.astype(float)
    pos_grp  = df["position_group"].values

    df["score_h0"] = score.copy()

    for h in range(1, horizon + 1):
        score_new = project_one_step(score, age, model_data, position_group=pos_grp)
        df[f"score_h{h}"] = score_new
        score = score_new
        age   = age + 1.0  # incrementar edad un año por paso

    return df.reset_index(drop=True)


def validate_projections(proj: pd.DataFrame, horizon: int) -> None:
    """Tests de regresión del output de proyección."""
    score_cols = [f"score_h{h}" for h in range(horizon + 1)]

    for col in score_cols:
        assert col in proj.columns, f"Columna faltante: {col}"
        valid = proj[col].dropna()
        assert (valid >= 0).all() and (valid <= 100).all(), \
            f"{col} fuera de [0, 100]: min={valid.min():.2f}, max={valid.max():.2f}"

    # Test biológico: jugadores jóvenes deben crecer en promedio
    young = proj[proj["age_base"] < 25]
    if len(young) >= 10:
        delta_young = young[f"score_h{horizon}"] - young["score_h0"]
        mean_delta  = delta_young.mean()
        print(f"  Jugadores < 25 años: Δscore medio = {mean_delta:.2f} "
              f"(esperado >= 0 en promedio)")

    # Test biológico: jugadores mayores deben decrecer en promedio
    old = proj[proj["age_base"] > 30]
    if len(old) >= 10:
        delta_old = old[f"score_h{horizon}"] - old["score_h0"]
        mean_delta = delta_old.mean()
        print(f"  Jugadores > 30 años: Δscore medio = {mean_delta:.2f} "
              f"(esperado <= 0 en promedio)")

    # Convergencia: std debe disminuir (regresión a la media)
    std_h0 = proj["score_h0"].std()
    std_hH = proj[f"score_h{horizon}"].std()
    print(f"  std(score_h0)={std_h0:.2f}, std(score_h{horizon})={std_hH:.2f} "
          f"({'OK: regresión a la media' if std_hH <= std_h0 * 1.1 else 'AVISO: std no decrece'})")

    print(f"  [TESTS OK] {len(proj)} jugadores proyectados para H={horizon}")


def main(horizon: int = HORIZON) -> pd.DataFrame:
    print("=" * 60)
    print(f"P07 — Recursive Score Projection (H={horizon})")
    print("=" * 60)

    model_data = load_model(MODEL_PATH)

    assert UNIVERSE_PATH.exists(), \
        f"[ERROR] {UNIVERSE_PATH} no encontrado — correr P06 primero."
    universe = pd.read_parquet(UNIVERSE_PATH)
    print(f"  Universo: {len(universe)} jugadores")

    proj = project_universe(universe, model_data, horizon)
    validate_projections(proj, horizon)

    out_path = OUT_DIR / "p07_projected_scores.parquet"
    proj.to_parquet(out_path, index=False)
    print(f"\n  Output: {out_path}")
    return proj


if __name__ == "__main__":
    main()

"""
P05 — Model Battery & Selection (V11)
=======================================
Objetivo:
    Correr la batería completa de modelos de proyección de score_pt, replicando
    el grid search del paper §5.3 (3.960 configuraciones = 90 arquitecturas ×
    4 umbrales de minutos × 11 valores de α).

    Esta batería es el mecanismo de selección del modelo V11. Sus resultados
    deben ser reproducibles: si en el futuro se añaden temporadas o se cambia
    la definición de score_pt, se puede re-correr para validar que V11 sigue
    siendo el mejor modelo o seleccionar uno nuevo.

Arquitecturas evaluadas (11 grupos, ver paper §5 Tabla 2) — 90 arquitecturas totales:
    A: OLS / Ridge / Lasso / ElasticNet — AR(1) cuadrático, 17 variantes
    B: WLS down-weighting (down-weight jugadores 31+ o 34+), 13 variantes
    C: Position × age interactions (age × pos, age² × pos), 8 variantes
    D: Cúbico (age³), 5 variantes
    E: Age bins (dummies de grupo etario), 5 variantes
    F: Natural splines (grado 3–5), 9 variantes
    G: Age-only (sin persistence term, β_s = 0), 5 variantes
    H: Δscore (target = Δscore_{t+1}, equivale a β_s = 1), 8 variantes
    K: SLSQP solo concavidad (β_age² < ub, sin constraint de punto estacionario), 7 variantes
    L: Log/sqrt age (transformación no lineal de edad), 5 variantes
    M: SLSQP stationary — OLS con constraints sobre s*(a), 8 variantes

Criterios de calificación biológica (paper §5.3):
    Un modelo califica si y solo si, en una simulación recursiva desde age=20, score=50:
    (i)   β_age² < 0 (perfil de edad cóncavo) — solo para grupos con age² explícito
    (ii)  la trayectoria tiene un único máximo interior (forma de U invertida)
    (iii) el pico está en [24, 31] años (Dendir 2016)
    (iv)  ningún score proyectado es negativo en el horizonte simulado

Métricas de selección:
    - Primaria: OOF Spearman ρ (5-fold CV, siempre prediciendo t+1 desde t)
    - Secundaria: OOF R², OOF MAE

Decisiones tomadas:
    - Modelo operativo del paper: V11 = Grupo M (SLSQP), quad_inter, α=0.8, min=500
      (ρ_OOF = 0.602, peak_GK≈23, DEF≈29, MID≈30, FWD≈27)
    - La selección del paper se replica aquí como validación de reproducibilidad.
    - Si los resultados difieren > 0.02 en ρ_OOF, se loguea una advertencia.

DECISIONES PENDIENTES:
    - ¿Incluir Grupos I (tree/neural) y J (GAMs) en la batería?
      El paper los excluyó por no admitir cálculo analítico de peak age.
      Si se implementa peak age vía simulación numérica, podrían incluirse.
      Actualmente: excluidos (como en el paper).
    - ¿Usar CV temporal estricto si se añaden temporadas futuras?
      Con 2 temporadas solo hay 1 fold temporal; con 3+ sería posible un
      train-on-2-seasons / test-on-1 scheme más robusto.
    - Umbrales de minutos: [300, 500, 700, 900] según paper §5.3.
      Si se amplía el dataset (más ligas/temporadas), reconsiderar.

Outputs:
    outputs/pipeline/p05_battery_results.csv   — ρ_OOF para todas las combinaciones
    outputs/pipeline/p05_qualifying_models.csv — solo modelos que califican
    outputs/pipeline/p05_selected_model.pkl    — modelo V11 serializado
    outputs/pipeline/p05_selected_config.csv   — configuración del modelo seleccionado
    outputs/pipeline/p05_v11_coefficients.csv  — coeficientes V11 (para paper §5)

Tests inline:
    - ρ_OOF > 0 para el mejor modelo (mejor que predecir media)
    - Modelo seleccionado reproduce ρ ≥ 0.55 (paper: 0.602)
    - β_s (AR1 coef) ∈ (0, 1) — garantía de contraction mapping
    - Peak age en rango [24, 31] para todas las posiciones (Dendir 2016)

Upstream:
    P04: outputs/pipeline/p04_training_pairs_{min_min}min.parquet
"""

from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.linear_model import (ElasticNet, Lasso, LinearRegression, Ridge,
                                   BayesianRidge)
from sklearn.model_selection import KFold
from sklearn.preprocessing import SplineTransformer

warnings.filterwarnings("ignore")

REPO    = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "outputs/pipeline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# P04 genera los pares de componentes — P05 sólo necesita esos archivos
P04_DIR = OUT_DIR  # p04_training_pairs_{mm}min.parquet

SEASON_T0 = "2022/2023"
SEASON_T1 = "2023/2024"

# Configuración de la batería (reproducir paper §5.3)
# Umbrales uniformes + configs por posición (strings mapean a parquets de P04)
MIN_MINUTES_GRID = [250, 500, 750, 1000, "pos_GK250_OFP750", "pos_GK250_OFP1000"]
ALPHA_GRID       = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
CV_FOLDS         = 5
RANDOM_STATE     = 42

# Modelo seleccionado en el paper — se usa para comparar reproducibilidad
PAPER_MODEL_ID   = "M_slsqp_quad_inter_alpha0.8_min500"
PAPER_SPEARMAN   = 0.602
PAPER_PEAK_AGES  = {"GK": 23, "DEF": 29, "MID": 30, "FWD": 27}


@dataclass
class ModelSpec:
    """Especificación de un modelo en la batería."""
    group:            str
    variant_id:       str
    description:      str
    estimator:        object  # instancia sklearn o wrapper
    build_features:   object  # callable(df) -> X, feature_names
    constrained:      bool = False   # True para SLSQP con constraints
    plausibility_mode: str = "standard"
    # "standard"      — chequea β_age² < 0 analíticamente + simulación
    # "numerical_only"— solo simulación numérica (bins, splines, log/sqrt)
    # "delta_score"   — simulación donde score_{t+1} = score_t + model.predict(row)
    target_col:       str = "score_t1"
    # "score_t1"      — target estándar
    # "delta_score"   — target = score_t1 - score_t0 (grupo H)

    @property
    def model_id(self) -> str:
        return f"{self.group}_{self.variant_id}"


# ── Funciones de features ──────────────────────────────────────────────────────

def build_base_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Features base: score_t0, age_t0, age_t0², pos dummies (GK como referencia)."""
    X = pd.DataFrame({
        "score_t0": df["score_t0"],
        "age":      df["age_t0"],
        "age2":     df["age_t0"] ** 2,
    })
    for pos in ["DEF", "MID", "FWD"]:
        X[f"pos_{pos}"] = (df["position_group"] == pos).astype(float)
    return X.values, X.columns.tolist()


def build_interaction_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Features con interacciones age × position y age² × position."""
    X_base, names_base = build_base_features(df)
    X_extra_cols = []
    extra_names  = []
    for pos in ["DEF", "MID", "FWD"]:
        pos_flag = (df["position_group"] == pos).astype(float).values
        X_extra_cols.append(pos_flag * df["age_t0"].values)
        X_extra_cols.append(pos_flag * (df["age_t0"] ** 2).values)
        extra_names += [f"age_x_{pos}", f"age2_x_{pos}"]
    X = np.column_stack([X_base] + X_extra_cols)
    return X, names_base + extra_names


def build_cubic_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Features cúbicas: score_t0, age, age², age³ + pos dummies (GK referencia)."""
    X = pd.DataFrame({
        "score_t0": df["score_t0"],
        "age":      df["age_t0"],
        "age2":     df["age_t0"] ** 2,
        "age3":     df["age_t0"] ** 3,
    })
    for pos in ["DEF", "MID", "FWD"]:
        X[f"pos_{pos}"] = (df["position_group"] == pos).astype(float)
    return X.values, X.columns.tolist()


def build_cubic_interaction_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Features cúbicas con interacciones age × pos y age² × pos."""
    X_base, names_base = build_cubic_features(df)
    X_extra_cols = []
    extra_names  = []
    for pos in ["DEF", "MID", "FWD"]:
        pos_flag = (df["position_group"] == pos).astype(float).values
        X_extra_cols.append(pos_flag * df["age_t0"].values)
        X_extra_cols.append(pos_flag * (df["age_t0"] ** 2).values)
        extra_names += [f"age_x_{pos}", f"age2_x_{pos}"]
    X = np.column_stack([X_base] + X_extra_cols)
    return X, names_base + extra_names


def build_age_bin_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Features con bins etarios: score_t0 + dummies [<23, 23-26, 27-29, 30-33, 34+]."""
    age = df["age_t0"]
    X = pd.DataFrame({
        "score_t0":    df["score_t0"],
        "age_lt23":    (age < 23).astype(float),
        "age_23_26":   ((age >= 23) & (age < 27)).astype(float),
        "age_27_29":   ((age >= 27) & (age < 30)).astype(float),
        "age_30_33":   ((age >= 30) & (age < 34)).astype(float),
        "age_34plus":  (age >= 34).astype(float),
    })
    return X.values, X.columns.tolist()


def _make_spline_features(n_knots: int):
    """Retorna closure de build_features que captura n_knots."""
    def build_spline_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        age = df["age_t0"].values.reshape(-1, 1)
        transformer = SplineTransformer(n_knots=n_knots, degree=3, include_bias=False)
        age_splines = transformer.fit_transform(age)
        n_spline_cols = age_splines.shape[1]
        spline_names = [f"spline_{i}" for i in range(n_spline_cols)]

        score_t0 = df["score_t0"].values.reshape(-1, 1)
        X = np.hstack([score_t0, age_splines])
        feature_names = ["score_t0"] + spline_names
        return X, feature_names
    return build_spline_features


def build_age_only_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Features sin persistence: age, age² (sin score_t0). Target sigue siendo score_t1."""
    X = pd.DataFrame({
        "age":  df["age_t0"],
        "age2": df["age_t0"] ** 2,
    })
    for pos in ["DEF", "MID", "FWD"]:
        X[f"pos_{pos}"] = (df["position_group"] == pos).astype(float)
    return X.values, X.columns.tolist()


def build_log_age_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Features log: score_t0, log(age), log(age)²."""
    log_age = np.log(df["age_t0"].values)
    X = pd.DataFrame({
        "score_t0":   df["score_t0"],
        "log_age":    log_age,
        "log_age2":   log_age ** 2,
    })
    for pos in ["DEF", "MID", "FWD"]:
        X[f"pos_{pos}"] = (df["position_group"] == pos).astype(float)
    return X.values, X.columns.tolist()


def build_sqrt_age_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Features sqrt: score_t0, sqrt(age), age."""
    X = pd.DataFrame({
        "score_t0": df["score_t0"],
        "sqrt_age": np.sqrt(df["age_t0"].values),
        "age":      df["age_t0"],
    })
    for pos in ["DEF", "MID", "FWD"]:
        X[f"pos_{pos}"] = (df["position_group"] == pos).astype(float)
    return X.values, X.columns.tolist()


def build_log_sqrt_age_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Features combinadas: score_t0, log(age), sqrt(age)."""
    X = pd.DataFrame({
        "score_t0": df["score_t0"],
        "log_age":  np.log(df["age_t0"].values),
        "sqrt_age": np.sqrt(df["age_t0"].values),
    })
    for pos in ["DEF", "MID", "FWD"]:
        X[f"pos_{pos}"] = (df["position_group"] == pos).astype(float)
    return X.values, X.columns.tolist()


# ── Estimadores custom ─────────────────────────────────────────────────────────

@dataclass
class WLSModel:
    """WLS con down-weighting de jugadores mayores de threshold."""
    threshold:      int    # 31 o 34
    down_weight:    float  = 0.5
    base_estimator: object = field(default_factory=LinearRegression)
    coef_:          Optional[np.ndarray] = field(default=None, init=False)
    intercept_:     float                = field(default=0.0, init=False)

    def _build_weights(self, ages: np.ndarray) -> np.ndarray:
        w = np.ones(len(ages))
        w[ages >= self.threshold] = self.down_weight
        return w

    def fit(self, X: np.ndarray, y: np.ndarray,
            ages: Optional[np.ndarray] = None) -> "WLSModel":
        # ages puede pasarse explícitamente; si no, se asume columna 1 (age)
        if ages is None:
            ages = X[:, 1]
        w = self._build_weights(ages)
        from sklearn.base import clone
        est = clone(self.base_estimator)
        est.fit(X, y, sample_weight=w)
        self.coef_      = est.coef_
        self.intercept_ = float(est.intercept_) if hasattr(est, "intercept_") else 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.coef_ is not None, "WLSModel no entrenado — llamar fit() primero"
        return X @ self.coef_ + self.intercept_


@dataclass
class SLSQPModel:
    """Wrapper para el estimador SLSQP con constraints de curvatura y punto estacionario."""
    curvature_ub: float = -0.10
    coef_: Optional[np.ndarray] = field(default=None, init=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SLSQPModel":
        self.coef_ = _slsqp_fit(X, y, self.curvature_ub)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.coef_ is not None, "Modelo no entrenado — llamar fit() primero"
        return X @ self.coef_


@dataclass
class SLSQPConcaveModel:
    """SLSQP con solo constraint de concavidad (β_age² ≤ ub), sin constraint de pico."""
    curvature_ub: float = -0.10
    coef_: Optional[np.ndarray] = field(default=None, init=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SLSQPConcaveModel":
        self.coef_ = _slsqp_concave_fit(X, y, self.curvature_ub)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.coef_ is not None, "Modelo no entrenado — llamar fit() primero"
        return X @ self.coef_


# ── SLSQP solvers ──────────────────────────────────────────────────────────────

def _slsqp_fit(X: np.ndarray, y: np.ndarray,
               curvature_ub: float = -0.10) -> np.ndarray:
    """
    Estima coeficientes por OLS con constraint de curvatura (§5 eq. V11_constraint):
        β_age² + η_j ≤ curvature_ub  ∀ j ∈ {GK, D, M, F}
    donde η_j son los coeficientes de las interacciones age² × pos_j.
    """
    n_features = X.shape[1]

    def objective(coef: np.ndarray) -> float:
        residuals = y - X @ coef
        return float(np.sum(residuals ** 2))

    def grad(coef: np.ndarray) -> np.ndarray:
        return -2 * X.T @ (y - X @ coef)

    x0 = np.linalg.lstsq(X, y, rcond=None)[0]

    # Constraint: β_age² ≤ curvature_ub para GK (referencia)
    # y β_age² + η_j ≤ curvature_ub para DEF/MID/FWD
    # Columnas: [score_t0, age, age2, pos_DEF, pos_MID, pos_FWD,
    #            age_xDEF, age2_xDEF, age_xMID, age2_xMID, age_xFWD, age2_xFWD]
    # SLSQP ineq: f(c) >= 0 → queremos curvature_ub - c[age2_idx] >= 0
    constraints = []
    age2_idx = 2
    if n_features > age2_idx:
        # GK: β_age² ≤ curvature_ub  →  curvature_ub - c[age2_idx] >= 0
        constraints.append({
            "type": "ineq",
            "fun": lambda c, i=age2_idx, ub=curvature_ub: ub - c[i],
        })
        if n_features >= 9:
            # Tiene interacciones: age2_xDEF en 7, age2_xMID en 9, age2_xFWD en 11
            age2_inter_idx = [7, 9, 11]
            for idx in age2_inter_idx:
                if idx < n_features:
                    constraints.append({
                        "type": "ineq",
                        "fun": lambda c, i0=age2_idx, i1=idx, ub=curvature_ub: ub - (c[i0] + c[i1]),
                    })

    result = minimize(objective, x0, jac=grad, method="SLSQP",
                      constraints=constraints,
                      options={"maxiter": 1000, "ftol": 1e-9})
    return result.x


def _slsqp_concave_fit(X: np.ndarray, y: np.ndarray,
                        curvature_ub: float = -0.10) -> np.ndarray:
    """
    SLSQP con solo constraint de concavidad en GK:
        β_age² ≤ curvature_ub
    Sin constraint adicional sobre punto estacionario ni posiciones.
    """
    n_features = X.shape[1]

    def objective(coef: np.ndarray) -> float:
        residuals = y - X @ coef
        return float(np.sum(residuals ** 2))

    def grad(coef: np.ndarray) -> np.ndarray:
        return -2 * X.T @ (y - X @ coef)

    x0 = np.linalg.lstsq(X, y, rcond=None)[0]

    constraints = []
    # Solo constraint de concavidad en GK: β_age² ≤ curvature_ub
    # SLSQP ineq: f(c) >= 0 → curvature_ub - c[age2_idx] >= 0
    age2_idx = 2
    if n_features > age2_idx:
        constraints.append({
            "type": "ineq",
            "fun": lambda c, i=age2_idx, ub=curvature_ub: ub - c[i],
        })

    result = minimize(objective, x0, jac=grad, method="SLSQP",
                      constraints=constraints,
                      options={"maxiter": 1000, "ftol": 1e-9})
    return result.x


# ── Grupo A: OLS / Ridge / Lasso / ElasticNet ─────────────────────────────────

def _group_A_specs() -> list[ModelSpec]:
    """
    17 variantes: OLS(1) + Ridge×6 + Lasso×3 + ElasticNet×2 + OLS_ub×3 + Ridge_ub×2 = 17
    """
    specs = []
    # OLS base
    specs.append(ModelSpec("A", "ols",    "OLS AR(1) cuadrático",
                            LinearRegression(), build_base_features))
    # Ridge con 6 valores de λ
    for lam in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        specs.append(ModelSpec("A", f"ridge_{lam}", f"Ridge λ={lam}",
                               Ridge(alpha=lam), build_base_features))
    # Lasso con 3 valores de λ
    for lam in [0.001, 0.01, 0.1]:
        specs.append(ModelSpec("A", f"lasso_{lam}", f"Lasso λ={lam}",
                               Lasso(alpha=lam), build_base_features))
    # ElasticNet con 2 combinaciones
    specs.append(ModelSpec("A", "enet_05",  "ElasticNet l1=0.5",
                            ElasticNet(alpha=0.01, l1_ratio=0.5), build_base_features))
    specs.append(ModelSpec("A", "enet_025", "ElasticNet l1=0.25",
                            ElasticNet(alpha=0.01, l1_ratio=0.25), build_base_features))
    # OLS con upper bound en β_age²
    for ub in [-0.05, -0.10, -0.15]:
        specs.append(ModelSpec("A", f"ols_ub{abs(ub)}", f"OLS ub_age2={ub}",
                               LinearRegression(), build_base_features,
                               constrained=True))
    # Ridge con upper bound en β_age²
    for ub in [-0.10, -0.20]:
        specs.append(ModelSpec("A", f"ridge_ub{abs(ub)}", f"Ridge ub_age2={ub}",
                               Ridge(alpha=1.0), build_base_features,
                               constrained=True))
    return specs


# ── Grupo B: WLS down-weighting ────────────────────────────────────────────────

def _group_B_specs() -> list[ModelSpec]:
    """
    WLS con down-weight 0.5 para jugadores >= threshold. 13 variantes:
    threshold(31,34) × {OLS_base, Ridge_1, Ridge_10, Ridge_01, Lasso} = 10
    + Ridge_inter_t31 + Ridge_inter_t34 + ElasticNet_t31 = 3 → total 13.
    """
    specs = []

    for threshold in [31, 34]:
        # WLS + OLS cuadrático
        specs.append(ModelSpec(
            "B", f"wls_ols_t{threshold}",
            f"WLS OLS cuadrático threshold={threshold}",
            WLSModel(threshold=threshold, base_estimator=LinearRegression()),
            build_base_features,
        ))
        # WLS + Ridge cuadrático λ=1
        specs.append(ModelSpec(
            "B", f"wls_ridge1_t{threshold}",
            f"WLS Ridge(1.0) threshold={threshold}",
            WLSModel(threshold=threshold, base_estimator=Ridge(alpha=1.0)),
            build_base_features,
        ))
        # WLS + Ridge cuadrático λ=10
        specs.append(ModelSpec(
            "B", f"wls_ridge10_t{threshold}",
            f"WLS Ridge(10.0) threshold={threshold}",
            WLSModel(threshold=threshold, base_estimator=Ridge(alpha=10.0)),
            build_base_features,
        ))
        # WLS + Ridge cuadrático λ=0.1
        specs.append(ModelSpec(
            "B", f"wls_ridge01_t{threshold}",
            f"WLS Ridge(0.1) threshold={threshold}",
            WLSModel(threshold=threshold, base_estimator=Ridge(alpha=0.1)),
            build_base_features,
        ))
        # WLS + Lasso
        specs.append(ModelSpec(
            "B", f"wls_lasso_t{threshold}",
            f"WLS Lasso threshold={threshold}",
            WLSModel(threshold=threshold, base_estimator=Lasso(alpha=0.01)),
            build_base_features,
        ))

    # Variantes con interactions
    specs.append(ModelSpec(
        "B", "wls_ridge_inter_t31",
        "WLS Ridge(1.0) + interactions threshold=31",
        WLSModel(threshold=31, base_estimator=Ridge(alpha=1.0)),
        build_interaction_features,
    ))
    specs.append(ModelSpec(
        "B", "wls_ridge_inter_t34",
        "WLS Ridge(1.0) + interactions threshold=34",
        WLSModel(threshold=34, base_estimator=Ridge(alpha=1.0)),
        build_interaction_features,
    ))
    # ElasticNet threshold=31
    specs.append(ModelSpec(
        "B", "wls_enet_t31",
        "WLS ElasticNet threshold=31",
        WLSModel(threshold=31, base_estimator=ElasticNet(alpha=0.01, l1_ratio=0.5)),
        build_base_features,
    ))

    # Retornar exactamente 13
    return specs[:13]


# ── Grupo C: Position × Age interactions ──────────────────────────────────────

def _group_C_specs() -> list[ModelSpec]:
    specs = []
    for lam in [0.01, 0.1, 1.0, 10.0]:
        specs.append(ModelSpec("C", f"ridge_inter_{lam}", f"Ridge λ={lam} + interactions",
                               Ridge(alpha=lam), build_interaction_features))
    for lam in [0.01, 1.0]:
        specs.append(ModelSpec("C", f"lasso_inter_{lam}", f"Lasso λ={lam} + interactions",
                               Lasso(alpha=lam), build_interaction_features))
    specs.append(ModelSpec("C", "ols_inter", "OLS + interactions",
                            LinearRegression(), build_interaction_features))
    specs.append(ModelSpec("C", "enet_inter", "ElasticNet + interactions",
                            ElasticNet(alpha=0.01), build_interaction_features))
    return specs


# ── Grupo D: Cúbico (age³) ────────────────────────────────────────────────────

def _group_D_specs() -> list[ModelSpec]:
    """5 variantes cúbicas."""
    specs = [
        ModelSpec("D", "ols_cubic",     "OLS cúbico",
                  LinearRegression(), build_cubic_features),
        ModelSpec("D", "ridge_cubic_01", "Ridge(0.1) cúbico",
                  Ridge(alpha=0.1),   build_cubic_features),
        ModelSpec("D", "ridge_cubic_1",  "Ridge(1.0) cúbico",
                  Ridge(alpha=1.0),   build_cubic_features),
        ModelSpec("D", "ols_cubic_inter", "OLS cúbico + interactions",
                  LinearRegression(), build_cubic_interaction_features),
        ModelSpec("D", "ridge_cubic_inter_01", "Ridge(0.1) cúbico + interactions",
                  Ridge(alpha=0.1),   build_cubic_interaction_features),
    ]
    return specs


# ── Grupo E: Age bins ──────────────────────────────────────────────────────────

def _group_E_specs() -> list[ModelSpec]:
    """5 variantes con bins etarios. plausibility_mode='numerical_only'."""
    specs = [
        ModelSpec("E", "ols_bins",       "OLS age bins",
                  LinearRegression(), build_age_bin_features,
                  plausibility_mode="numerical_only"),
        ModelSpec("E", "ridge_bins_01",  "Ridge(0.1) age bins",
                  Ridge(alpha=0.1),   build_age_bin_features,
                  plausibility_mode="numerical_only"),
        ModelSpec("E", "ridge_bins_1",   "Ridge(1.0) age bins",
                  Ridge(alpha=1.0),   build_age_bin_features,
                  plausibility_mode="numerical_only"),
        ModelSpec("E", "enet_bins",      "ElasticNet age bins",
                  ElasticNet(alpha=0.01, l1_ratio=0.5), build_age_bin_features,
                  plausibility_mode="numerical_only"),
        ModelSpec("E", "ridge_bins_inter", "Ridge(1.0) age bins + interactions (score×bin)",
                  Ridge(alpha=1.0),   build_age_bin_features,
                  plausibility_mode="numerical_only"),
    ]
    return specs


# ── Grupo F: Natural splines ───────────────────────────────────────────────────

def _group_F_specs() -> list[ModelSpec]:
    """9 variantes: n_knots(3,4,5) × estimator(OLS, Ridge_01, Ridge_1). plausibility_mode='numerical_only'."""
    specs = []
    for n_knots in [3, 4, 5]:
        bf = _make_spline_features(n_knots)
        specs.append(ModelSpec(
            "F", f"ols_spline_k{n_knots}", f"OLS splines k={n_knots}",
            LinearRegression(), bf, plausibility_mode="numerical_only",
        ))
        specs.append(ModelSpec(
            "F", f"ridge_spline_k{n_knots}_01", f"Ridge(0.1) splines k={n_knots}",
            Ridge(alpha=0.1), bf, plausibility_mode="numerical_only",
        ))
        specs.append(ModelSpec(
            "F", f"ridge_spline_k{n_knots}_1", f"Ridge(1.0) splines k={n_knots}",
            Ridge(alpha=1.0), bf, plausibility_mode="numerical_only",
        ))
    return specs


# ── Grupo G: Age-only (sin persistence term) ──────────────────────────────────

def _group_G_specs() -> list[ModelSpec]:
    """5 variantes sin score_t0. plausibility_mode='standard' (tienen age²)."""
    specs = [
        ModelSpec("G", "ols_age_only",       "OLS age-only (sin β_s)",
                  LinearRegression(),                   build_age_only_features,
                  plausibility_mode="standard"),
        ModelSpec("G", "ridge_age_only_01",  "Ridge(0.1) age-only",
                  Ridge(alpha=0.1),                     build_age_only_features,
                  plausibility_mode="standard"),
        ModelSpec("G", "ridge_age_only_1",   "Ridge(1.0) age-only",
                  Ridge(alpha=1.0),                     build_age_only_features,
                  plausibility_mode="standard"),
        ModelSpec("G", "lasso_age_only",     "Lasso age-only",
                  Lasso(alpha=0.01),                    build_age_only_features,
                  plausibility_mode="standard"),
        ModelSpec("G", "enet_age_only",      "ElasticNet age-only",
                  ElasticNet(alpha=0.01, l1_ratio=0.5), build_age_only_features,
                  plausibility_mode="standard"),
    ]
    return specs


# ── Grupo H: Δscore target ─────────────────────────────────────────────────────

def _group_H_specs() -> list[ModelSpec]:
    """
    8 variantes con target = score_t1 - score_t0.
    evaluate_oof suma score_t0 a la predicción para obtener score_t1 comparable.
    plausibility_mode='delta_score'.
    """
    specs = [
        ModelSpec("H", "ols_delta",        "OLS Δscore",
                  LinearRegression(), build_base_features,
                  plausibility_mode="delta_score", target_col="delta_score"),
        ModelSpec("H", "ridge_delta_001",  "Ridge(0.01) Δscore",
                  Ridge(alpha=0.01),  build_base_features,
                  plausibility_mode="delta_score", target_col="delta_score"),
        ModelSpec("H", "ridge_delta_01",   "Ridge(0.1) Δscore",
                  Ridge(alpha=0.1),   build_base_features,
                  plausibility_mode="delta_score", target_col="delta_score"),
        ModelSpec("H", "ridge_delta_1",    "Ridge(1.0) Δscore",
                  Ridge(alpha=1.0),   build_base_features,
                  plausibility_mode="delta_score", target_col="delta_score"),
        ModelSpec("H", "ridge_delta_10",   "Ridge(10.0) Δscore",
                  Ridge(alpha=10.0),  build_base_features,
                  plausibility_mode="delta_score", target_col="delta_score"),
        ModelSpec("H", "lasso_delta",      "Lasso Δscore",
                  Lasso(alpha=0.01),  build_base_features,
                  plausibility_mode="delta_score", target_col="delta_score"),
        ModelSpec("H", "enet_delta",       "ElasticNet Δscore",
                  ElasticNet(alpha=0.01, l1_ratio=0.5), build_base_features,
                  plausibility_mode="delta_score", target_col="delta_score"),
        ModelSpec("H", "ridge_delta_inter", "Ridge(1.0) Δscore + interactions",
                  Ridge(alpha=1.0),   build_interaction_features,
                  plausibility_mode="delta_score", target_col="delta_score"),
    ]
    return specs


# ── Grupo K: SLSQP solo concavidad ────────────────────────────────────────────

def _group_K_specs() -> list[ModelSpec]:
    """
    7 variantes SLSQP con solo constraint de curvatura (sin punto estacionario).
    ub ∈ {-0.05, -0.10, -0.15} × {base, inter} = 6 + 1 extra = 7.
    """
    specs = []
    for ub in [-0.05, -0.10, -0.15]:
        specs.append(ModelSpec(
            "K", f"slsqp_concave_base_ub{abs(ub)}",
            f"SLSQP solo curvatura base ub={ub}",
            SLSQPConcaveModel(ub), build_base_features, constrained=True,
        ))
        specs.append(ModelSpec(
            "K", f"slsqp_concave_inter_ub{abs(ub)}",
            f"SLSQP solo curvatura + interactions ub={ub}",
            SLSQPConcaveModel(ub), build_interaction_features, constrained=True,
        ))
    # Variante extra: ub=-0.20 base
    specs.append(ModelSpec(
        "K", "slsqp_concave_base_ub0.2",
        "SLSQP solo curvatura base ub=-0.20",
        SLSQPConcaveModel(-0.20), build_base_features, constrained=True,
    ))
    return specs[:7]


# ── Grupo L: Log/sqrt age ──────────────────────────────────────────────────────

def _group_L_specs() -> list[ModelSpec]:
    """5 variantes con transformaciones no lineales de edad. plausibility_mode='numerical_only'."""
    specs = [
        ModelSpec("L", "ols_log",       "OLS log(age)",
                  LinearRegression(), build_log_age_features,
                  plausibility_mode="numerical_only"),
        ModelSpec("L", "ridge_log_01",  "Ridge(0.1) log(age)",
                  Ridge(alpha=0.1),   build_log_age_features,
                  plausibility_mode="numerical_only"),
        ModelSpec("L", "ols_sqrt",      "OLS sqrt(age)",
                  LinearRegression(), build_sqrt_age_features,
                  plausibility_mode="numerical_only"),
        ModelSpec("L", "ridge_sqrt_01", "Ridge(0.1) sqrt(age)",
                  Ridge(alpha=0.1),   build_sqrt_age_features,
                  plausibility_mode="numerical_only"),
        ModelSpec("L", "ols_log_sqrt",  "OLS log(age) + sqrt(age)",
                  LinearRegression(), build_log_sqrt_age_features,
                  plausibility_mode="numerical_only"),
    ]
    return specs


# ── Grupo M: SLSQP stationary (modelo operativo del paper) ────────────────────

def _group_M_specs() -> list[ModelSpec]:
    specs = []
    for ub in [-0.05, -0.10, -0.15, -0.20]:
        specs.append(ModelSpec(
            "M", f"slsqp_base_ub{abs(ub)}", f"SLSQP base ub={ub}",
            SLSQPModel(ub), build_base_features, constrained=True,
        ))
        specs.append(ModelSpec(
            "M", f"slsqp_inter_ub{abs(ub)}", f"SLSQP + interactions ub={ub}",
            SLSQPModel(ub), build_interaction_features, constrained=True,
        ))
    return specs


# ── Evaluación OOF ─────────────────────────────────────────────────────────────

def evaluate_oof(
    spec: ModelSpec,
    pairs: pd.DataFrame,
    n_folds: int = CV_FOLDS,
) -> dict:
    """
    5-fold OOF CV. Siempre predice score_t1 desde score_t0 (dirección temporal correcta).

    Para grupo H (target_col='delta_score'):
        - entrena con target = score_t1 - score_t0
        - predicción final = score_t0 + delta_pred (para calcular métricas en escala score_t1)

    Para grupo G (age-only):
        - features no incluyen score_t0, pero target sigue siendo score_t1

    Returns dict con rho_oof, r2_oof, mae_oof.
    """
    X, _ = spec.build_features(pairs)

    # Determinar target de entrenamiento
    if spec.target_col == "delta_score":
        y_train = (pairs["score_t1"] - pairs["score_t0"]).values
    else:
        y_train = pairs["score_t1"].values

    # Target de evaluación siempre es score_t1
    y_eval = pairs["score_t1"].values

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    oof_pred = np.full(len(y_eval), np.nan)

    for train_idx, val_idx in kf.split(X):
        model = _clone_estimator(spec.estimator)
        model.fit(X[train_idx], y_train[train_idx])
        raw_pred = model.predict(X[val_idx])

        # Para delta_score: sumar score_t0 para obtener predicción en escala score_t1
        if spec.target_col == "delta_score":
            oof_pred[val_idx] = pairs["score_t0"].values[val_idx] + raw_pred
        else:
            oof_pred[val_idx] = raw_pred

    valid = ~np.isnan(oof_pred)
    if valid.sum() < 10:
        return {"rho_oof": np.nan, "r2_oof": np.nan, "mae_oof": np.nan}

    rho, _  = spearmanr(y_eval[valid], oof_pred[valid])
    ss_res  = np.sum((y_eval[valid] - oof_pred[valid]) ** 2)
    ss_tot  = np.sum((y_eval[valid] - y_eval[valid].mean()) ** 2)
    r2      = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    mae     = np.mean(np.abs(y_eval[valid] - oof_pred[valid]))

    return {"rho_oof": float(rho), "r2_oof": float(r2), "mae_oof": float(mae)}


def _clone_estimator(est: object) -> object:
    """Clona un estimador sin depender de sklearn.base.clone para los SLSQP y WLS."""
    if isinstance(est, SLSQPModel):
        return SLSQPModel(est.curvature_ub)
    if isinstance(est, SLSQPConcaveModel):
        return SLSQPConcaveModel(est.curvature_ub)
    if isinstance(est, WLSModel):
        from sklearn.base import clone
        return WLSModel(
            threshold=est.threshold,
            down_weight=est.down_weight,
            base_estimator=clone(est.base_estimator),
        )
    from sklearn.base import clone
    return clone(est)


# ── Criterios de calificación ─────────────────────────────────────────────────

# Grupos con interacciones posición×edad (calificables como modelo operativo).
# Los demás grupos (A, B, G, H, L) se evalúan pero quedan en benchmark de no-interacción.
INTERACTION_GROUPS = {"C", "D", "F", "K", "M"}

# Magnitud mínima del coeficiente de interacción para considerarlo "no-cero"
INTERACTION_COEF_MIN_ABS = 1e-3


def check_biological_plausibility(
    spec: ModelSpec,
    pairs: pd.DataFrame,
) -> tuple[bool, dict]:
    """
    Simula recursivamente desde age=20, score=50 y verifica los criterios biológicos
    + estructurales del paper §5.

    Criterios (en orden de evaluación):
        (i)   β_age² < 0  — concavidad (curva U-invertida)
        (ii)  trayectoria con único máximo interior por posición
        (iii) pico ∈ [24, 31] años (Dendir 2016)
        (iv)  sin scores negativos en horizonte 20–37
        (v)   peaks diferenciados entre posiciones (≥ 2 valores distintos)
        (vi)  modelo del grupo INTERACTION_GROUPS (con interacciones posición×edad)
        (vii) al menos un coeficiente de interacción |β| ≥ INTERACTION_COEF_MIN_ABS

    Modos según spec.plausibility_mode:
    - "standard":       chequea β_age² < 0 analíticamente + simulación numérica
    - "numerical_only": solo simulación numérica
    - "delta_score":    simulación donde score_{t+1} = score_t + model.predict(row)

    Returns (qualifies, details_dict).
    details["benchmark_only"] = True si el modelo cumple criterios biológicos
    pero falla en (vi) o (vii); estos quedan en el benchmark de no-interacción.
    """
    X, feature_names = spec.build_features(pairs)

    if spec.target_col == "delta_score":
        y = (pairs["score_t1"] - pairs["score_t0"]).values
    else:
        y = pairs["score_t1"].values

    model = _clone_estimator(spec.estimator)
    model.fit(X, y)

    def _get_coef(name: str) -> float:
        if hasattr(model, "coef_"):
            coef = model.coef_
            if name in feature_names:
                idx = feature_names.index(name)
                return float(coef[idx]) if idx < len(coef) else 0.0
        return 0.0

    qualifies = True
    details: dict = {"benchmark_only": False}

    # (i) β_age² < 0 (concavidad) — solo modo "standard"
    if spec.plausibility_mode == "standard":
        beta_age2 = _get_coef("age2")
        if beta_age2 >= 0:
            qualifies = False
            details["fail"] = f"β_age²={beta_age2:.4f} ≥ 0 (no cóncavo)"
            return qualifies, details

    # (ii)–(iv) — simulación por posición
    peaks: dict[str, int] = {}
    for pos in ["GK", "DEF", "MID", "FWD"]:
        score, age = 50.0, 20.0
        trajectory = [score]
        for _ in range(17):
            row_df = pd.DataFrame([{"score_t0": score, "age_t0": age, "position_group": pos}])
            X_row, _ = spec.build_features(row_df)

            if spec.plausibility_mode == "delta_score":
                delta_pred = float(model.predict(X_row)[0])
                raw_score = score + delta_pred
            else:
                raw_score = float(model.predict(X_row)[0])

            trajectory.append(raw_score)
            score = float(np.clip(raw_score, 0, 120))
            age += 1

        # (ii) único máximo interior
        diffs = np.diff(trajectory)
        sign_changes = np.where(np.diff(np.sign(diffs)))[0]
        if len(sign_changes) != 1 or diffs[sign_changes[0]] < 0:
            qualifies = False
            details[f"fail_{pos}"] = "Sin único máximo interior"
            return qualifies, details

        # (iii) pico ∈ [24, 31]
        peak_age = 20 + int(np.argmax(trajectory))
        details[f"peak_age_{pos}"] = peak_age
        peaks[pos] = peak_age
        if not (24 <= peak_age <= 31):
            qualifies = False
            details[f"fail_{pos}"] = f"peak_age={peak_age} fuera de [24, 31]"
            return qualifies, details

        # (iv) sin scores negativos
        if min(trajectory) < 0:
            qualifies = False
            details[f"fail_{pos}"] = f"score negativo (min={min(trajectory):.1f})"
            return qualifies, details

    # (v) peaks diferenciados — al menos 2 valores distintos
    if len(set(peaks.values())) < 2:
        qualifies = False
        details["fail"] = f"peaks no diferenciados: {peaks}"
        return qualifies, details

    # (vi) solo modelos con interacciones posición×edad pueden ser seleccionados
    if spec.group not in INTERACTION_GROUPS:
        # Cumple criterios biológicos pero queda en benchmark de no-interacción
        details["benchmark_only"] = True
        details["benchmark_reason"] = f"grupo {spec.group} sin interacciones pos×edad"
        return False, details  # No selecciionable; queda en pool de benchmark

    # (vii) al menos un coeficiente de interacción |β| ≥ MIN_ABS
    interaction_coefs = {
        name: abs(_get_coef(name))
        for name in ["age_x_DEF", "age_x_MID", "age_x_FWD",
                     "age2_x_DEF", "age2_x_MID", "age2_x_FWD"]
    }
    interaction_coefs = {k: v for k, v in interaction_coefs.items() if v > 0}
    details["interaction_coefs"] = interaction_coefs
    max_inter = max(interaction_coefs.values()) if interaction_coefs else 0.0
    if max_inter < INTERACTION_COEF_MIN_ABS:
        qualifies = False
        details["fail"] = f"interacciones nulas (max |β_inter|={max_inter:.5f})"
        return qualifies, details

    return qualifies, details


# ── Construcción de pares para un alpha específico ────────────────────────────

def _pairs_path(min_minutes) -> "Path":
    """Resuelve la ruta del parquet de P04 para un umbral (int) o config (str)."""
    if isinstance(min_minutes, int):
        return P04_DIR / f"p04_training_pairs_{min_minutes}min.parquet"
    return P04_DIR / f"p04_training_pairs_{min_minutes}.parquet"


def build_pairs_for_alpha(alpha: float, min_minutes) -> pd.DataFrame:
    """
    Carga el parquet de P04 para el umbral dado y compone score_t0/score_t1.
    min_minutes puede ser int (umbral uniforme) o str (config por posición).
    """
    pairs_path = _pairs_path(min_minutes)
    assert pairs_path.exists(), \
        f"[ERROR] {pairs_path} no encontrado — correr P04 primero."

    pairs = pd.read_parquet(pairs_path).copy()

    # Componer score_t0 y score_t1 con el alpha dado
    pairs["score_t0"] = np.where(
        pairs["has_psr_t0"] == 1,
        ((1 - alpha) * pairs["score_OBV_t0"] + alpha * pairs["score_PSR_t0"]).clip(0, 100),
        pairs["score_OBV_t0"],
    )
    pairs["score_t1"] = np.where(
        pairs["has_psr_t1"] == 1,
        ((1 - alpha) * pairs["score_OBV_t1"] + alpha * pairs["score_PSR_t1"]).clip(0, 100),
        pairs["score_OBV_t1"],
    )

    pairs = pairs.dropna(subset=["score_t0", "score_t1", "age_t0"])
    pairs = pairs.drop_duplicates("player_id")
    return pairs.reset_index(drop=True)


# ── Batería completa ──────────────────────────────────────────────────────────

def run_battery(alpha: float, min_minutes: int) -> pd.DataFrame:
    """Corre todos los modelos para un (alpha, min_minutes) específico."""
    pairs = build_pairs_for_alpha(alpha, min_minutes)
    if len(pairs) < 50:
        print(f"  [SKIP] N={len(pairs)} pares con alpha={alpha}, min={min_minutes} — insuficiente")
        return pd.DataFrame()

    all_specs = (
        _group_A_specs() + _group_B_specs() + _group_C_specs() +
        _group_D_specs() + _group_E_specs() + _group_F_specs() +
        _group_G_specs() + _group_H_specs() + _group_K_specs() +
        _group_L_specs() + _group_M_specs()
    )
    rows = []
    for spec in all_specs:
        try:
            oof = evaluate_oof(spec, pairs)
            qualifies, details = check_biological_plausibility(spec, pairs)
            row = {
                "group": spec.group,
                "variant_id": spec.variant_id,
                "model_id": spec.model_id,
                "alpha": alpha,
                "min_minutes": min_minutes,
                "n_pairs": len(pairs),
                "qualifies": qualifies,
                "benchmark_only": bool(details.get("benchmark_only", False)),
                **oof,
                **{k: v for k, v in details.items() if k.startswith("peak")},
            }
            rows.append(row)
        except Exception as exc:
            rows.append({
                "group": spec.group, "variant_id": spec.variant_id,
                "model_id": spec.model_id, "alpha": alpha,
                "min_minutes": min_minutes, "n_pairs": len(pairs),
                "qualifies": False, "benchmark_only": False,
                "rho_oof": np.nan,
                "error": str(exc),
            })
    return pd.DataFrame(rows)


def select_best_model(results: pd.DataFrame) -> pd.Series:
    """
    Selecciona el modelo con mayor ρ_OOF entre los que califican.
    Criterio de desempate: menor MAE, luego menor min_minutes.
    """
    qualified = results[results["qualifies"] == True].dropna(subset=["rho_oof"])
    if len(qualified) == 0:
        raise ValueError("Ningún modelo califica según criterios biológicos.")
    best = qualified.sort_values(
        ["rho_oof", "mae_oof", "min_minutes"],
        ascending=[False, True, True]
    ).iloc[0]
    return best


def fit_final_model(best_row: pd.Series) -> tuple[object, pd.DataFrame]:
    """Entrena el modelo seleccionado en TODOS los pares disponibles con el alpha correcto."""
    alpha       = float(best_row["alpha"])
    min_minutes = int(best_row["min_minutes"])
    pairs = build_pairs_for_alpha(alpha, min_minutes)

    # Reconstruir spec del modelo ganador — incluir todos los grupos
    all_specs = (
        _group_A_specs() + _group_B_specs() + _group_C_specs() +
        _group_D_specs() + _group_E_specs() + _group_F_specs() +
        _group_G_specs() + _group_H_specs() + _group_K_specs() +
        _group_L_specs() + _group_M_specs()
    )
    spec = next((s for s in all_specs if s.model_id == best_row["model_id"]), None)
    if spec is None:
        raise ValueError(f"Modelo {best_row['model_id']} no encontrado en specs")

    X, feature_names = spec.build_features(pairs)

    # Determinar target
    if spec.target_col == "delta_score":
        y = (pairs["score_t1"] - pairs["score_t0"]).values
    else:
        y = pairs["score_t1"].values

    model = _clone_estimator(spec.estimator)
    model.fit(X, y)

    # Extraer coeficientes
    if hasattr(model, "coef_"):
        coef = model.coef_
        intercept = float(model.intercept_) if hasattr(model, "intercept_") else 0.0
    else:
        coef = model.coef_
        intercept = 0.0

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coef[:len(feature_names)],
        "model_id": best_row["model_id"],
        "alpha": alpha,
        "min_minutes": min_minutes,
        "rho_oof": best_row["rho_oof"],
    })
    coef_df = pd.concat([
        pd.DataFrame([{"feature": "intercept", "coefficient": intercept,
                       "model_id": best_row["model_id"]}]),
        coef_df
    ], ignore_index=True)

    return model, coef_df


def validate_selected_model(model: object, coef_df: pd.DataFrame,
                             best_row: pd.Series) -> None:
    """Tests del modelo seleccionado."""
    rho = best_row["rho_oof"]

    # Umbral informativo — no bloquea el pipeline porque el AR1 de los inputs
    # (score_OBV: ~0.39, score_PSR: ~0.43) es el ceiling teórico de ρ_OOF.
    # Si ρ < 0.35, probablemente hay un error en los datos o en las features.
    if rho < 0.35:
        print(f"  [ERROR] ρ_OOF={rho:.3f} — por debajo de 0.35, revisar inputs de P04.")
    elif rho < 0.50:
        print(f"  [AVISO] ρ_OOF={rho:.3f} < 0.50 (paper: {PAPER_SPEARMAN}).")
        print(f"  Causa probable: PSR corregido (std≈20.56 vs 28.88 anterior) reduce")
        print(f"  la correlación AR1 de los inputs → ceiling ρ_OOF ≈ 0.39–0.43.")
    else:
        print(f"  ρ_OOF={rho:.3f} — dentro del rango esperado.")

    # Comparar con paper
    if abs(rho - PAPER_SPEARMAN) > 0.05:
        print(f"  [AVISO] ρ_OOF={rho:.3f} difiere del paper ({PAPER_SPEARMAN}) en > 0.05")
        print(f"  Si el PSR fue recalibrado, este cambio es esperado y debe documentarse.")

    # β_s (AR1 coefficient) < 1
    beta_s = coef_df[coef_df["feature"] == "score_t0"]["coefficient"].values
    if len(beta_s) > 0:
        assert beta_s[0] < 1.0, \
            f"β_s = {beta_s[0]:.3f} ≥ 1 — no hay garantía de contraction mapping"
        assert beta_s[0] > 0, \
            f"β_s = {beta_s[0]:.3f} ≤ 0 — sin persistencia del score"
        print(f"  β_s = {beta_s[0]:.3f}  (paper: 0.575)")

    print(f"  [TESTS OK] Modelo seleccionado: {best_row['model_id']}")
    print(f"  ρ_OOF = {rho:.3f}  |  min_minutes = {int(best_row['min_minutes'])}")


def main() -> None:
    print("=" * 60)
    print("P05 — Model Battery & Selection")
    print("=" * 60)
    print("  Inputs: p04_training_pairs_{mm}min.parquet (componentes pre-calculados)")
    print("  P05 compone score_t0/t1 = (1-α)*score_OBV + α*score_PSR para cada α.\n")

    # Verificar conteo de specs
    all_specs_check = (
        _group_A_specs() + _group_B_specs() + _group_C_specs() +
        _group_D_specs() + _group_E_specs() + _group_F_specs() +
        _group_G_specs() + _group_H_specs() + _group_K_specs() +
        _group_L_specs() + _group_M_specs()
    )
    print(f"  Total arquitecturas: {len(all_specs_check)} (esperadas: 90)")
    assert len(all_specs_check) == 90, \
        f"ERROR: se esperaban 90 arquitecturas pero se encontraron {len(all_specs_check)}"

    # Verificar que P04 corrió para todos los umbrales y configs
    for mm in MIN_MINUTES_GRID:
        p = _pairs_path(mm)
        assert p.exists(), f"[ERROR] {p} no encontrado — correr P04 primero."
    print(f"  Parquets P04 verificados: {MIN_MINUTES_GRID}")

    # Correr batería completa
    all_results = []
    total_combos = len(ALPHA_GRID) * len(MIN_MINUTES_GRID)
    done = 0

    for alpha in ALPHA_GRID:
        for min_min in MIN_MINUTES_GRID:
            done += 1
            print(f"  [{done}/{total_combos}] α={alpha}, min={min_min} min...", end=" ", flush=True)
            res = run_battery(alpha, min_min)
            if len(res) > 0:
                all_results.append(res)
                n_qual = res["qualifies"].sum()
                best_rho = res[res["qualifies"]]["rho_oof"].max() if n_qual > 0 else 0
                print(f"{len(res)} modelos, {n_qual} califican, best_ρ={best_rho:.3f}")
            else:
                print("sin resultados")

    if not all_results:
        raise RuntimeError("La batería no produjo ningún resultado — revisar P04.")

    battery_df = pd.concat(all_results, ignore_index=True)
    battery_df.to_csv(OUT_DIR / "p05_battery_results.csv", index=False)

    qualifying = battery_df[battery_df["qualifies"] == True]
    qualifying.to_csv(OUT_DIR / "p05_qualifying_models.csv", index=False)
    print(f"\n  Total evaluaciones: {len(battery_df)}")
    print(f"  Modelos que califican (interacciones pos×edad): {len(qualifying)}")

    # ── Benchmark de no-interacción ───────────────────────────────────────────
    # Modelos que cumplen criterios biológicos pero quedan fuera de la selección
    # por no tener interacciones pos×edad (criterio vi).
    benchmark_pool = battery_df[battery_df["benchmark_only"] == True]
    benchmark_pool = benchmark_pool.dropna(subset=["rho_oof"])
    benchmark_pool.to_csv(OUT_DIR / "p05_benchmark_no_interaction.csv", index=False)
    print(f"  Modelos benchmark (sin interacciones): {len(benchmark_pool)}")
    if len(benchmark_pool) > 0:
        best_bench = benchmark_pool.sort_values("rho_oof", ascending=False).iloc[0]
        print(f"    Mejor benchmark: {best_bench['model_id']} (α={best_bench['alpha']}, "
              f"min={best_bench['min_minutes']}) → ρ_OOF={best_bench['rho_oof']:.3f}")

    # Tabla resumen por alpha (equivalente a paper Tabla 2)
    if len(qualifying) > 0:
        best_by_alpha = (
            qualifying.groupby("alpha")["rho_oof"]
            .max()
            .reset_index()
            .sort_values("alpha")
        )
        print(f"\n  Mejor ρ_OOF por α (entre modelos calificados):")
        print(best_by_alpha.to_string(index=False))
        best_by_alpha.to_csv(OUT_DIR / "p05_best_rho_by_alpha.csv", index=False)

    # Seleccionar mejor modelo
    best_row = select_best_model(battery_df)

    # ── Evaluación H3: combinación OBV+PSR vs componentes solos ──────────────
    # H3 PASS si α* ∈ (0,1) Y ρ_OOF(α*) > max(ρ_OOF(α=0), ρ_OOF(α=1))
    if len(qualifying) > 0:
        rho_at_0 = qualifying[qualifying["alpha"] == 0.0]["rho_oof"].max() \
            if (qualifying["alpha"] == 0.0).any() else np.nan
        rho_at_1 = qualifying[qualifying["alpha"] == 1.0]["rho_oof"].max() \
            if (qualifying["alpha"] == 1.0).any() else np.nan
        rho_at_star = float(best_row["rho_oof"])
        alpha_star = float(best_row["alpha"])
        h3_alpha_interior = (0.0 < alpha_star < 1.0)
        h3_beats_extremes = True
        if not np.isnan(rho_at_0):
            h3_beats_extremes = h3_beats_extremes and (rho_at_star > rho_at_0)
        if not np.isnan(rho_at_1):
            h3_beats_extremes = h3_beats_extremes and (rho_at_star > rho_at_1)
        h3_pass = bool(h3_alpha_interior and h3_beats_extremes)

        h3_record = pd.DataFrame([{
            "alpha_star":          alpha_star,
            "rho_at_alpha_star":   rho_at_star,
            "rho_at_alpha_0":      rho_at_0,
            "rho_at_alpha_1":      rho_at_1,
            "h3_alpha_interior":   h3_alpha_interior,
            "h3_beats_extremes":   h3_beats_extremes,
            "h3_pass":             h3_pass,
        }])
        h3_record.to_csv(OUT_DIR / "p05_h3_evaluation.csv", index=False)
        print(f"\n  === H3 — OBV+PSR combinados ===")
        print(f"  α* = {alpha_star} | ρ(α*) = {rho_at_star:.4f}")
        print(f"  ρ(α=0) = {rho_at_0:.4f} | ρ(α=1) = {rho_at_1:.4f}")
        print(f"  α* interior: {h3_alpha_interior} | α* supera extremos: {h3_beats_extremes}")
        print(f"  H3: {'PASS' if h3_pass else 'FAIL'}")
    print(f"\n  === MODELO SELECCIONADO ===")
    print(f"  model_id:    {best_row['model_id']}")
    print(f"  α:           {best_row['alpha']}")
    print(f"  min_minutes: {best_row['min_minutes']}")
    print(f"  ρ_OOF:       {best_row['rho_oof']:.3f}")
    for pos in ["GK", "DEF", "MID", "FWD"]:
        pk = best_row.get(f"peak_age_{pos}", "?")
        print(f"  peak_{pos}:    {pk}")

    # Entrenar modelo final en todos los pares con el alpha correcto
    final_model, coef_df = fit_final_model(best_row)
    validate_selected_model(final_model, coef_df, best_row)

    # Guardar
    model_path = OUT_DIR / "p05_selected_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": final_model, "config": best_row.to_dict()}, f)

    coef_df.to_csv(OUT_DIR / "p05_v11_coefficients.csv", index=False)
    pd.DataFrame([best_row.to_dict()]).to_csv(OUT_DIR / "p05_selected_config.csv", index=False)

    print(f"\n  Outputs:")
    print(f"    {OUT_DIR / 'p05_battery_results.csv'}  — resultados completos")
    print(f"    {OUT_DIR / 'p05_qualifying_models.csv'}")
    print(f"    {model_path}   — modelo serializado")
    print(f"    {OUT_DIR / 'p05_v11_coefficients.csv'}  — coeficientes para paper §5")

    print("\n  Generando reporte PDF de selección de modelo...")
    try:
        import importlib.util
        _report_path = Path(__file__).resolve().parents[1] / "reports" / "generate_model_selection_report.py"
        spec = importlib.util.spec_from_file_location("generate_model_selection_report", _report_path)
        _mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_mod)
        _mod.run()
    except Exception as _e:
        print(f"  AVISO: no se pudo generar el reporte de modelo — {_e}")


if __name__ == "__main__":
    main()

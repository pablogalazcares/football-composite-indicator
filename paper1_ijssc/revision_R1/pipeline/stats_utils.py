"""Utilidades estadísticas del análisis R1 (PREREGISTRO §2–§3).

Convención: `pos` es el grupo de la temporada de origen. ρ_W = promedio ponderado por n de los Spearman dentro de cada
grupo. El bootstrap remuestrea jugadores con reemplazo dentro de cada grupo (estratificado), con índices compartidos
para las comparaciones pareadas.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan
    return float(spearmanr(x, y)[0])


def rho_within(x, y, pos, groups) -> tuple[float, dict]:
    x, y, pos = np.asarray(x, float), np.asarray(y, float), np.asarray(pos)
    per, w = {}, {}
    for g in groups:
        m = pos == g
        per[g] = spearman(x[m], y[m]) if m.sum() >= 5 else np.nan
        w[g] = m.sum()
    vals = np.array([per[g] for g in groups])
    ws = np.array([w[g] for g in groups], float)
    ok = ~np.isnan(vals)
    return (float(np.sum(vals[ok] * ws[ok]) / ws[ok].sum()) if ok.any() else np.nan), per


def stratified_indices(pos, groups, rng) -> np.ndarray:
    pos = np.asarray(pos)
    idx = []
    for g in groups:
        ii = np.flatnonzero(pos == g)
        if len(ii):
            idx.append(rng.choice(ii, size=len(ii), replace=True))
    return np.concatenate(idx)


def bootstrap(stat: Callable[[np.ndarray], dict], pos, groups, B: int, seed: int) -> pd.DataFrame:
    """Aplica `stat(idx) -> dict de valores` a B remuestreos estratificados. Devuelve un DataFrame (B × claves)."""
    rng = np.random.default_rng(seed)
    rows = [stat(stratified_indices(pos, groups, rng)) for _ in range(B)]
    return pd.DataFrame(rows)


def ci(series: pd.Series) -> list[float]:
    s = pd.Series(series).dropna()
    return [float(s.quantile(0.025)), float(s.quantile(0.975))] if len(s) else [np.nan, np.nan]


def top_quartile_precision(pred, obs, pos, groups, q: float = 0.75) -> float:
    pred, obs, pos = np.asarray(pred, float), np.asarray(obs, float), np.asarray(pos)
    num, den = 0, 0
    for g in groups:
        m = pos == g
        if m.sum() < 8:
            continue
        pt = pred[m] >= np.quantile(pred[m], q)
        ot = obs[m] >= np.quantile(obs[m], q)
        num += int((pt & ot).sum())
        den += int(pt.sum())
    return num / den if den else np.nan


def weighted_rank_corr(x, y, w) -> float:
    rx, ry = rankdata(x), rankdata(y)
    w = np.asarray(w, float)
    mx, my = np.average(rx, weights=w), np.average(ry, weights=w)
    cov = np.average((rx - mx) * (ry - my), weights=w)
    return float(cov / np.sqrt(np.average((rx - mx) ** 2, weights=w) * np.average((ry - my) ** 2, weights=w)))


def weighted_rho_within(x, y, w, pos, groups) -> float:
    x, y, w, pos = map(np.asarray, (x, y, w, pos))
    vals, ns = [], []
    for g in groups:
        m = pos == g
        if m.sum() >= 5:
            vals.append(weighted_rank_corr(x[m], y[m], w[m]))
            ns.append(m.sum())
    return float(np.average(vals, weights=ns)) if vals else np.nan


def partial_spearman(x, y, covars: np.ndarray) -> float:
    """Correlación de los residuos de rank(x) y rank(y) sobre las covariables (OLS con intercepto)."""
    rx, ry = rankdata(x), rankdata(y)
    Z = np.column_stack([np.ones(len(rx)), covars])
    bx, *_ = np.linalg.lstsq(Z, rx, rcond=None)
    by, *_ = np.linalg.lstsq(Z, ry, rcond=None)
    ex, ey = rx - Z @ bx, ry - Z @ by
    return float(np.corrcoef(ex, ey)[0, 1])


def auc(score, label) -> float:
    score, label = np.asarray(score, float), np.asarray(label, bool)
    if label.all() or (~label).all():
        return np.nan
    r = rankdata(score)
    n1, n0 = label.sum(), (~label).sum()
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

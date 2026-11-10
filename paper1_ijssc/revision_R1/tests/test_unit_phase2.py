"""Pruebas unitarias sintéticas de la Fase 2 (rápidas)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

import r06_models as R6
import stats_utils as SU

G = ("DEF", "MID", "FWD")


def _df(n=60, seed=0):
    rng = np.random.default_rng(seed)
    pos = np.array(G * (n // 3))
    s = rng.uniform(20, 80, n)
    age = rng.uniform(19, 34, n)
    y = 20 + 0.6 * s + rng.normal(0, 8, n)
    return pd.DataFrame({"s_t": s, "age_t": age, "pos": pos, "s_tk": y})


def test_rho_within_weighted_average():
    d = _df()
    w, per = SU.rho_within(d.s_t, d.s_tk, d.pos, G)
    assert w == pytest.approx(np.mean([per[g] for g in G]))  # tamaños iguales -> promedio simple


def test_bootstrap_is_deterministic_and_stratified():
    d = _df()
    f = lambda ii: {"n_def": int((d.pos.values[ii] == "DEF").sum())}
    a = SU.bootstrap(f, d.pos, G, 50, 7)
    b = SU.bootstrap(f, d.pos, G, 50, 7)
    assert a.equals(b)
    assert (a.n_def == 20).all()


def test_auc_matches_sklearn():
    rng = np.random.default_rng(1)
    s, l = rng.normal(size=200), rng.random(200) > 0.4
    assert SU.auc(s, l) == pytest.approx(roc_auc_score(l, s))


def test_m1_is_rank_equivalent_to_m0_within_position():
    """PREREGISTRO P2: dentro de posición, M1 es monótono en s_t, así que su ρ dentro de posición es el de M0."""
    d = _df(90, 3)
    f = R6.fit_all(d, G, 26.0)
    p0, p1 = f["M0"].predict(d.s_t, d.age_t, d.pos), f["M1"].predict(d.s_t, d.age_t, d.pos)
    r0, _ = SU.rho_within(p0, d.s_tk, d.pos, G)
    r1, _ = SU.rho_within(p1, d.s_tk, d.pos, G)
    assert r0 == pytest.approx(r1)
    imp, _ = SU.rho_within(f["M1"].project(d.s_t, d.age_t, d.pos, 3), d.s_t, d.pos, G)
    assert imp == pytest.approx(1.0)


def test_recursive_projection_matches_manual():
    d = _df(30, 4)
    m = R6.Fitted("M2", G, 26.0).fit(d.s_t, d.age_t, d.pos, d.s_tk)
    two = m.project(d.s_t, d.age_t, d.pos, 2)
    manual = m.predict(m.predict(d.s_t, d.age_t, d.pos), d.age_t + 1, d.pos)
    assert np.allclose(two, manual)


def test_top_quartile_precision_perfect_and_partial_spearman():
    rng = np.random.default_rng(5)
    x = rng.normal(size=80)
    pos = np.array(G * 26 + ("DEF", "MID"))
    assert SU.top_quartile_precision(x, x, pos, G) == pytest.approx(1.0)
    z = rng.normal(size=80)
    y = x + 2 * z
    assert abs(SU.partial_spearman(x, y, z[:, None])) > abs(SU.spearman(x, y)) - 1  # sanity: finite
    assert -1 <= SU.partial_spearman(x, y, z[:, None]) <= 1

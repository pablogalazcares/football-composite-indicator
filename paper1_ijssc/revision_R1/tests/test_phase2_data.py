"""Pruebas de integración de la Fase 2 (slow)."""
from __future__ import annotations

import json
import subprocess

import numpy as np
import pandas as pd
import pytest

import common
import r06_models as R6

pytestmark = pytest.mark.slow
SNAP = common.R1 / "tests" / "_snapshots" / "r1_key_numbers.json"


def test_models_fitted_only_on_training_transition(cfg):
    """§5.4: los coeficientes salen solo de A→B; armar los pares por otra vía da lo mismo."""
    ps = pd.read_parquet(cfg.out("players_season_full.parquet"))
    a, b = ps[(ps.season == "A") & ps.eligible_main], ps[(ps.season == "B") & ps.eligible_main]
    d = a.merge(b[["player_id", "score_obv_ar1"]], on="player_id", suffixes=("", "_tk"))
    d = d[d.position_group_season.isin(common.OUTFIELD)]
    d = pd.DataFrame({"s_t": d.score_obv_ar1, "age_t": d.age_jan1, "pos": d.position_group_season, "s_tk": d.score_obv_ar1_tk})
    f_direct = R6.Fitted("M2", common.OUTFIELD, 26.0).fit(d.s_t, d.age_t, d.pos, d.s_tk)
    fits = json.loads(cfg.out("model_fits.json").read_text())["fits"]["M2"]
    assert f_direct.coef[0] == pytest.approx(fits["s_t"])
    assert f_direct.intercept == pytest.approx(fits["intercept"])
    assert len(d) == json.loads(cfg.out("model_fits.json").read_text())["n_train"]


def test_submitted_rho_k_reproduced(cfg):
    imp = json.loads(cfg.out("implied.json").read_text())
    assert np.allclose(imp["submitted_rho_k_reproduced"]["pooled"], [0.9402, 0.7204, 0.4193, 0.2045, 0.0839], atol=1e-4)


def test_preregistration_committed_before_metrics():
    """G2: el commit del preregistro precede a cualquier commit de métricas entre temporadas."""
    def first_commit(path):
        out = subprocess.run(["git", "log", "--format=%ct", "--reverse", "--", path], cwd=common.REPO,
                             capture_output=True, text=True, timeout=120).stdout.split()
        return int(out[0]) if out else None
    pre = first_commit("paper1_ijssc/revision_R1/PREREGISTRO.md")
    assert pre is not None
    met = first_commit("paper1_ijssc/revision_R1/outputs/r1/metrics_main.json")
    assert met is None or met >= pre


def test_no_placeholder_or_nan_in_numbers(cfg):
    num = json.loads(cfg.res("paper_numbers.json").read_text())
    bad = [k for k, v in num.items() if v is None or (isinstance(v, float) and np.isnan(v))]
    assert not bad, bad[:10]
    tex = cfg.res("numbers.tex").read_text()
    assert "NA}" not in tex


def test_key_numbers_snapshot(cfg):
    num = json.loads(cfg.res("paper_numbers.json").read_text())
    if not SNAP.exists():
        pytest.skip("snapshot no creado todavía")
    snap = json.loads(SNAP.read_text())
    diff = {k: (v, num.get(k)) for k, v in snap.items() if not np.isclose(v, num.get(k, np.nan), atol=1e-9)}
    assert not diff, diff

"""Pruebas unitarias sintéticas de la Fase 1 (rápidas)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import r02_minutes as R2
import r04_indicator as R4


def _match(subs_rows, red_rows=None, ends=(2700.0, 2900.0)):
    sxi = pd.DataFrame({"match_id": 1, "team_id": 1, "team": "X", "player_id": [10, 11, 12],
                        "player_name": list("abc"), "position": "Center Back"})
    subs = pd.DataFrame(subs_rows, columns=["match_id", "team_id", "period", "t_s", "off_id", "on_id", "on_name"])
    pend = pd.DataFrame({"match_id": 1, "period": [1, 2], "end_s": list(ends)})
    ev = pd.DataFrame({"match_id": 1, "player_id": [10, 11, 12, 13], "team": "X",
                       "card": [None, None, None, None], "period": 1, "t_s": 1.0})
    if red_rows:
        ev = pd.concat([ev, pd.DataFrame(red_rows)], ignore_index=True)
    return R2.match_minutes(ev, sxi, subs, pend).set_index("player_id").minutes


def test_minutes_full_sub_and_red():
    m = _match([[1, 1, 2, 600.0, 11, 13, "d"]],
               red_rows=[{"match_id": 1, "player_id": 12, "team": "X", "card": "Red Card", "period": 1, "t_s": 1800.0}])
    assert m[10] == pytest.approx((2700 + 2900) / 60)        # titular completo, con descuento
    assert m[11] == pytest.approx((2700 + 600) / 60)         # sale a los 10' del segundo tiempo
    assert m[13] == pytest.approx((2900 - 600) / 60)         # entra a los 10' del segundo tiempo
    assert m[12] == pytest.approx(1800 / 60)                 # roja a los 30'


def test_minutes_second_yellow_and_halftime_sub():
    m = _match([[1, 1, 2, 0.0, 10, 13, "d"]],
               red_rows=[{"match_id": 1, "player_id": 11, "team": "X", "card": "Second Yellow", "period": 2,
                          "t_s": 300.0}])
    assert m[10] == pytest.approx(2700 / 60)
    assert m[13] == pytest.approx(2900 / 60)
    assert m[11] == pytest.approx((2700 + 300) / 60)


def test_clock_to_elapsed():
    assert R2._clock_to_elapsed("00:46:59.077", 1) == pytest.approx(46 * 60 + 59.077)
    assert R2._clock_to_elapsed("01:18:42.369", 2) == pytest.approx(33 * 60 + 42.369)


def _agg(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in ("A", "B"):
        for pid in range(40):
            for et in ("Pass", "Carry", "Shot", "Duel"):
                rows.append({"player_id": pid, "season": s, "event_type": et, "n_events": int(rng.integers(1, 50)),
                             "obv_for_mean": rng.normal(), "obv_against_mean": rng.normal()})
    agg = pd.DataFrame(rows)
    pos = agg[["player_id", "season"]].drop_duplicates()
    pos["position_group"] = np.where(pos.player_id % 2 == 0, "MID", "FWD")
    return agg, pos


W_FOR = {"MID": {"Pass": 0.5, "Carry": 0.4, "Shot": 0.05}, "FWD": {"Pass": 0.5, "Shot": 0.1}}
W_AG = {"MID": {"Duel": 0.3, "Carry": 0.3}, "FWD": {"Duel": 0.3}}
MACRO = {"MID": {"w_for": 0.53, "w_against": 0.47}, "FWD": {"w_for": 0.83, "w_against": 0.17}}


def test_percentiles_within_season_are_isolated():
    """REQ-C5: alterar los datos de otra temporada no cambia el score de esta."""
    agg, pos = _agg()
    s1 = R4.score_obv(agg, pos, W_FOR, W_AG, MACRO).set_index(["player_id", "season"]).score
    agg2 = agg.copy()
    mask = agg2.season == "B"
    agg2.loc[mask, "obv_for_mean"] = agg2.loc[mask, "obv_for_mean"].sample(frac=1, random_state=3).values * 10
    s2 = R4.score_obv(agg2, pos, W_FOR, W_AG, MACRO).set_index(["player_id", "season"]).score
    a = s1.xs("A", level="season")
    b = s2.xs("A", level="season")
    assert np.allclose(a.sort_index(), b.sort_index())


def test_score_bounds_and_missing_component():
    agg, pos = _agg(1)
    s = R4.score_obv(agg, pos, W_FOR, W_AG, MACRO)
    assert s.score.between(0, 100).all()
    # sin eventos AGAINST para FWD se usa 50 en ese componente
    agg_f = agg[~((agg.event_type == "Duel") & (agg.player_id % 2 == 1))]
    s2 = R4.score_obv(agg_f, pos, W_FOR, W_AG, MACRO)
    fwd = s2[s2.position_group == "FWD"]
    assert fwd.obv_against_pt.isna().all()
    assert np.allclose(fwd.score, 0.83 * fwd.obv_for_pt + 0.17 * 50)

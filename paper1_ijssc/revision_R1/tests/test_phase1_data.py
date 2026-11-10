"""Pruebas de integración de la Fase 1 sobre los artefactos reales (slow)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import common
import r04_indicator as R4

pytestmark = pytest.mark.slow


def _j(cfg, name):
    return json.loads(cfg.out(name).read_text())


def test_minutes_validation_meets_req_d2(cfg):
    v = _j(cfg, "minutes_validation.json")
    lim = cfg.raw["validation"]
    for season in ("A", "B"):
        assert v[season]["median_abs_err_min"] <= lim["minutes_median_abs_err_max"]
        assert v[season]["season_total_spearman"] >= lim["minutes_season_total_rho_min"]
        assert v[season]["only_in_events"] == 0 and v[season]["only_in_lineups"] == 0


def test_legacy_mode_reproduces_submitted_p03(cfg):
    c = _j(cfg, "indicator_checks.json")
    assert c["raw_vs_p02_rows_matched"] == c["raw_vs_p02_rows_p02"]
    assert c["raw_vs_p02_max_abs_diff_for"] == 0 and c["raw_vs_p02_max_abs_diff_against"] == 0
    assert c["raw_vs_p02_n_events_equal"]
    assert c["legacy_n_compared"] == 1100
    assert c["legacy_max_abs_diff_vs_p03"] <= 1e-9
    assert c["legacy_nan_pattern_equal"]


def test_ar1_weights_reestimated_from_raw_on_AB_match_constants(cfg):
    c = _j(cfg, "indicator_checks.json")
    assert c["ar1_reestimated_max_abs_diff_vs_constants"] <= 5e-4
    assert c["ar1_reestimated_event_sets_equal"]


def test_revision_scores_of_AB_do_not_depend_on_C(cfg):
    """Anti-fuga (REQ-C2/C5): quitar C no cambia el indicador de A y B."""
    agg = pd.read_parquet(cfg.out("obv_agg_player_season_type.parquet"))
    ps = pd.read_parquet(cfg.out("players_season.parquet"))
    pos = ps[["player_id", "season", "position_group_season"]].rename(columns={"position_group_season": "position_group"})
    pool = ps[ps.minutes_covered > 0][["player_id", "season"]]
    agg = agg.merge(pool, on=["player_id", "season"])
    P3 = common.p03()
    full = R4.score_obv(agg, pos, P3.AR1_WEIGHTS_FOR, P3.AR1_WEIGHTS_AGAINST, P3.MACRO_WEIGHTS)
    noC = R4.score_obv(agg[agg.season != "C"], pos, P3.AR1_WEIGHTS_FOR, P3.AR1_WEIGHTS_AGAINST, P3.MACRO_WEIGHTS)
    a = full[full.season != "C"].set_index(["player_id", "season"]).score.sort_index()
    b = noC.set_index(["player_id", "season"]).score.sort_index()
    assert np.allclose(a, b, equal_nan=True)


def test_identity_unique_and_display_names(cfg):
    ps = pd.read_parquet(cfg.out("players_season.parquet"))
    assert not ps.duplicated(["player_id", "season"]).any()
    assert ps.groupby("player_id").display_name.nunique().max() == 1
    names = ps.drop_duplicates("player_id").set_index("player_id").display_name
    assert names[4127] == "David Raya"
    assert names[3669] == "Virgil van Dijk"
    assert names[3272] == "Oriol Romeu"
    assert names[3531] == "Mohamed Salah"


def test_ages_and_positions_coverage(cfg):
    df = pd.read_parquet(cfg.out("players_season_full.parquet"))
    el = df[df.eligible_main]
    assert el.age_jan1.isna().mean() <= 0.01
    assert el.position_group_season.notna().all()
    assert el.age_jan1.between(15, 45).all()


def test_transitions_schema_and_panel_single_base(cfg):
    tr = pd.read_parquet(cfg.out("transitions.parquet"))
    need = {"player_id", "t", "tk", "k", "score_obv_ar1_t", "score_obv_ar1_tk", "age_jan1_t",
            "position_group_season_t", "position_group_season_tk", "eligible_main_t", "eligible_main_tk", "present_tk"}
    assert need <= set(tr.columns)
    assert not tr.duplicated(["player_id", "t", "tk"]).any()
    panel = pd.read_parquet(cfg.out("panel_universe.parquet"))
    assert panel.season.nunique() == 1 and panel.season.iloc[0] == cfg.labels[-1]  # REQ-D8


def test_no_event_level_data_inside_revision_folder(cfg):
    """REQ-R6: ningún parquet/CSV a nivel evento dentro de revision_R1/."""
    bad = []
    for f in list(common.R1.rglob("*.parquet")) + list(common.R1.rglob("*.csv")):
        try:
            cols = pd.read_parquet(f).columns if f.suffix == ".parquet" else pd.read_csv(f, nrows=5).columns
        except Exception:
            continue
        if {"obv_for_net", "under_pressure"} & set(cols) or "timestamp" in cols:
            bad.append(str(f))
    assert not bad, bad
    assert not cfg.paths["cache"].is_relative_to(common.REPO)

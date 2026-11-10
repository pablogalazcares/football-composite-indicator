from __future__ import annotations

import pytest

import common


def test_config_loads_and_seasons(cfg):
    assert cfg.labels == ["A", "B", "C"]
    assert cfg.season("B").season_id == 281
    assert cfg.transitions(1) == [("A", "B"), ("B", "C")]
    assert cfg.transitions(2) == [("A", "C")]
    assert cfg.seed == 20261108


def test_paths_resolve_inside_repo_or_cache(cfg):
    assert cfg.paths["raw_events"].is_relative_to(common.REPO)
    assert cfg.paths["outputs"].is_relative_to(common.R1)
    assert not cfg.paths["cache"].is_relative_to(common.REPO)  # D-17


def test_adding_season_d_extends_transitions():
    c = common.load_config()
    raw = c.raw | {"seasons": c.raw["seasons"] + [{"label": "D", "season_id": 999, "name": "2025/2026",
                                                    "jan1": "2026-01-01"}]}
    seasons = tuple(common.Season(**s) for s in raw["seasons"])
    c2 = common.Config(raw=raw, seasons=seasons, paths=c.paths)
    assert c2.transitions(3) == [("A", "D")]
    assert ("C", "D") in c2.transitions(1)


def test_training_transition_must_be_first_two():
    c = common.load_config()
    raw = dict(c.raw)
    raw["design"] = dict(raw["design"], training_transition=["B", "C"])
    with pytest.raises(ValueError):
        common.Config(raw=raw, seasons=c.seasons, paths=c.paths)


def test_pos_group_mapping():
    assert common.pos_group("Right Center Back") == "DEF"
    assert common.pos_group("Center Attacking Midfield") == "MID"
    assert common.pos_group("Left Wing") == "FWD"
    assert common.pos_group("Goalkeeper") == "GK"
    assert common.pos_group(None) is None

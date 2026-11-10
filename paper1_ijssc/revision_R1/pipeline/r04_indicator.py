"""R04 — Indicador OBV dentro de posición (REQ-C1…C5).

Receta (idéntica a P03 enviado):
  1. Media por ocurrencia de obv_for_net y obv_against_net por (jugador, temporada, tipo de evento), sobre los
     eventos con OBV.
  2. Shrinkage bayesiano K = 10 de la media FOR de Shot hacia la media de (temporada, grupo).
  3. Percentil (rank promedio) dentro de (temporada, grupo, tipo) de la media FOR y de −AGAINST.
  4. Promedio ponderado por los pesos AR1 de P03 (eventos con AR1 ≥ 0,10; Shot FOR forzado en MID/FWD).
  5. Mezcla macro FOR/AGAINST por grupo; si falta un componente se usa 50.

Modos:
  legacy  : universo = jugadores de p02 enviado, grupo canónico P01 (un grupo por jugador). Reproduce P03.
  revision: universo = jugadores con minutos y eventos OBV en la temporada, grupo de la temporada (D-06).
Variantes en modo revisión: pesos AR1 (principal), pesos iguales (REQ-C3), macro 0,5/0,5, posición canónica.
Salidas: outputs/r1/indicator_player_season.parquet, outputs/r1/obv_agg_player_season_type.parquet,
         outputs/r1/indicator_checks.json.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr

from common import Config, load_config, p03


def aggregate_obv(cfg: Config) -> pd.DataFrame:
    parts = []
    for s in cfg.seasons:
        ev = pd.read_parquet(cfg.cache(f"events_{s.label}.parquet"),
                             columns=["player_id", "type", "obv_for_net", "obv_against_net"])
        ev = ev.dropna(subset=["obv_for_net"])
        g = ev.groupby(["player_id", "type"]).agg(n_events=("obv_for_net", "size"),
                                                  obv_for_mean=("obv_for_net", "mean"),
                                                  obv_against_mean=("obv_against_net", "mean")).reset_index()
        parts.append(g.assign(season=s.label))
    return pd.concat(parts, ignore_index=True).rename(columns={"type": "event_type"})


def _weight_table(table: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows = []
    for pos, w in table.items():
        tot = sum(w.values())
        rows += [{"position_group": pos, "event_type": e, "w": v / tot} for e, v in w.items()]
    return pd.DataFrame(rows)


def score_obv(agg: pd.DataFrame, positions: pd.DataFrame, w_for: dict, w_ag: dict, macro: dict,
              k_shot: int = 10) -> pd.DataFrame:
    """agg: player_id, season, event_type, n_events, obv_for_mean, obv_against_mean.
    positions: player_id, season, position_group (un grupo por jugador-temporada)."""
    df = agg.merge(positions, on=["player_id", "season"], how="left")
    shot = df.event_type == "Shot"
    mu = df[shot].groupby(["season", "position_group"])["obv_for_mean"].transform("mean")
    n = df.loc[shot, "n_events"].clip(lower=0)
    df.loc[shot, "obv_for_mean"] = n / (n + k_shot) * df.loc[shot, "obv_for_mean"] + k_shot / (n + k_shot) * mu
    keys = ["season", "position_group", "event_type"]
    df["pct_for"] = df.groupby(keys)["obv_for_mean"].rank(method="average", pct=True) * 100
    df["pct_ag"] = (-df.obv_against_mean).groupby([df[k] for k in keys]).rank(method="average", pct=True) * 100

    def wmean(col: str, table: dict) -> pd.Series:
        t = df.merge(_weight_table(table), on=["position_group", "event_type"], how="inner").dropna(subset=[col])
        t["pw"] = t[col] * t.w
        g = t.groupby(["player_id", "season"])
        return g["pw"].sum() / g["w"].sum()

    out = df.drop_duplicates(["player_id", "season"])[["player_id", "season", "position_group"]].set_index(
        ["player_id", "season"])
    out["obv_for_pt"] = wmean("pct_for", w_for)
    out["obv_against_pt"] = wmean("pct_ag", w_ag)
    wf = out.position_group.map(lambda p: macro.get(p, macro["MID"])["w_for"])
    wa = out.position_group.map(lambda p: macro.get(p, macro["MID"])["w_against"])
    both_na = out.obv_for_pt.isna() & out.obv_against_pt.isna()
    out["score"] = wf * out.obv_for_pt.fillna(50) + wa * out.obv_against_pt.fillna(50)
    out.loc[both_na | out.position_group.isna(), "score"] = np.nan
    return out.reset_index()


def reestimate_ar1(agg: pd.DataFrame, positions: pd.DataFrame, a: str, b: str) -> tuple[dict, dict]:
    """Réplica de P02b + regla de selección de P03, estimada SOLO con la transición a→b (REQ-C2)."""
    df = agg.merge(positions, on=["player_id", "season"], how="left").dropna(subset=["position_group"])
    x, y = df[df.season == a], df[df.season == b]
    w_for, w_ag = {}, {}
    for pos in ["GK", "DEF", "MID", "FWD"]:
        w_for[pos], w_ag[pos] = {}, {}
        for et in sorted(df.event_type.unique()):
            m = x[(x.position_group == pos) & (x.event_type == et)].merge(
                y[(y.position_group == pos) & (y.event_type == et)], on="player_id")
            if len(m) < 10:
                continue
            rf = spearmanr(m.obv_for_mean_x, m.obv_for_mean_y)[0]
            ra = spearmanr(-m.obv_against_mean_x, -m.obv_against_mean_y)[0]
            if pos == "GK" and et not in ("Pass", "Carry", "Goal Keeper"):
                continue
            if np.isfinite(rf) and (rf >= 0.10 or (et == "Shot" and pos in ("MID", "FWD"))):
                w_for[pos][et] = float(rf)
            if np.isfinite(ra) and ra >= 0.10 and et != "Shot":
                w_ag[pos][et] = float(ra)
    return w_for, w_ag


def run(cfg: Config) -> dict:
    P3 = p03()
    W_FOR, W_AG, MACRO = P3.AR1_WEIGHTS_FOR, P3.AR1_WEIGHTS_AGAINST, P3.MACRO_WEIGHTS
    K = int(cfg.raw["indicator"]["shot_shrinkage_k"])
    agg = aggregate_obv(cfg)
    agg.to_parquet(cfg.out("obv_agg_player_season_type.parquet"), index=False)
    checks: dict = {}
    smap = {s.label: s.name for s in cfg.seasons}

    # ── modo legacy (REQ-C1) ─────────────────────────────────────────────────
    p02 = pd.read_parquet(cfg.paths["submitted_pipeline"] / "p02_obv_player_season.parquet")
    p02["season"] = p02.season_name.map({v: k for k, v in smap.items()})
    m = agg.merge(p02[["player_id", "season", "event_type", "n_events", "obv_for_mean", "obv_against_mean"]],
                  on=["player_id", "season", "event_type"], how="inner", suffixes=("", "_p02"))
    checks["raw_vs_p02_rows_matched"] = int(len(m))
    checks["raw_vs_p02_rows_p02"] = int(len(p02))
    checks["raw_vs_p02_max_abs_diff_for"] = float((m.obv_for_mean - m.obv_for_mean_p02).abs().max())
    checks["raw_vs_p02_max_abs_diff_against"] = float((m.obv_against_mean - m.obv_against_mean_p02).abs().max())
    checks["raw_vs_p02_n_events_equal"] = bool((m.n_events == m.n_events_p02).all())
    canon = pd.read_parquet(cfg.paths["submitted_pipeline"] / "p01_canonical_positions.parquet")
    leg_ids = p02[["player_id", "season"]].drop_duplicates()
    agg_leg = agg.merge(leg_ids, on=["player_id", "season"])
    pos_leg = leg_ids.merge(canon[["player_id", "position_group"]], on="player_id", how="left")
    leg = score_obv(agg_leg, pos_leg, W_FOR, W_AG, MACRO, K)
    p03s = pd.read_parquet(cfg.paths["submitted_pipeline"] / "p03_score_player_season.parquet")
    p03s["season"] = p03s.season_name.map({v: k for k, v in smap.items()})
    cmp = leg.merge(p03s[["player_id", "season", "score_OBV"]], on=["player_id", "season"])
    both = cmp.dropna(subset=["score", "score_OBV"])
    checks["legacy_n_compared"] = int(len(both))
    checks["legacy_max_abs_diff_vs_p03"] = float((both.score - both.score_OBV).abs().max())
    checks["legacy_nan_pattern_equal"] = bool((cmp.score.isna() == cmp.score_OBV.isna()).all())
    # pesos AR1 re-estimados desde crudo con A→B en modo legacy (deben reproducir las constantes)
    wf, wa = reestimate_ar1(agg_leg, pos_leg, "A", "B")
    diffs = [abs(wf[p].get(e, np.nan) - v) for p in W_FOR for e, v in W_FOR[p].items()] + \
            [abs(wa[p].get(e, np.nan) - v) for p in W_AG for e, v in W_AG[p].items()]
    checks["ar1_reestimated_max_abs_diff_vs_constants"] = float(np.nanmax(diffs))
    checks["ar1_reestimated_event_sets_equal"] = all(set(wf[p]) == set(W_FOR[p]) and set(wa[p]) == set(W_AG[p])
                                                    for p in W_FOR)

    # ── modo revisión ────────────────────────────────────────────────────────
    ps = pd.read_parquet(cfg.out("players_season.parquet"))
    pool = ps[ps.minutes_covered > 0][["player_id", "season"]]
    agg_rev = agg.merge(pool, on=["player_id", "season"])
    pos_rev = ps[["player_id", "season", "position_group_season"]].rename(
        columns={"position_group_season": "position_group"})
    pos_can = ps[["player_id", "season", "position_group_canonical", "position_group_season"]].copy()
    pos_can["position_group"] = pos_can.position_group_canonical.fillna(pos_can.position_group_season)
    pos_can = pos_can[["player_id", "season", "position_group"]]
    equal_for = {p: {e: 1.0 for e in W_FOR[p]} for p in W_FOR}
    equal_ag = {p: {e: 1.0 for e in W_AG[p]} for p in W_AG}
    macro05 = {p: {"w_for": 0.5, "w_against": 0.5} for p in MACRO}
    main = score_obv(agg_rev, pos_rev, W_FOR, W_AG, MACRO, K).rename(columns={"score": "score_obv_ar1"})
    variants = {
        "score_obv_equal": score_obv(agg_rev, pos_rev, equal_for, equal_ag, MACRO, K),
        "score_obv_ar1_macro05": score_obv(agg_rev, pos_rev, W_FOR, W_AG, macro05, K),
        "score_obv_ar1_canonpos": score_obv(agg_rev, pos_can, W_FOR, W_AG, MACRO, K),
    }
    out = main[["player_id", "season", "position_group", "obv_for_pt", "obv_against_pt", "score_obv_ar1"]]
    for name, v in variants.items():
        out = out.merge(v[["player_id", "season", "score"]].rename(columns={"score": name}),
                        on=["player_id", "season"], how="left")
    out = out.merge(leg[["player_id", "season", "score"]].rename(columns={"score": "score_obv_legacy"}),
                    on=["player_id", "season"], how="left")
    nobv = agg_rev.groupby(["player_id", "season"])["n_events"].sum().rename("n_obv_events").reset_index()
    out = out.merge(nobv, on=["player_id", "season"], how="left")
    out.to_parquet(cfg.out("indicator_player_season.parquet"), index=False)
    # comparación descriptiva modo revisión frente a legacy en A/B (mismo jugador-temporada)
    j = out.dropna(subset=["score_obv_ar1", "score_obv_legacy"])
    checks["revision_vs_legacy_AB_n"] = int(len(j))
    checks["revision_vs_legacy_AB_spearman"] = float(spearmanr(j.score_obv_ar1, j.score_obv_legacy)[0])
    checks["revision_vs_legacy_AB_max_abs_diff"] = float((j.score_obv_ar1 - j.score_obv_legacy).abs().max())
    checks["revision_n_by_season"] = out.groupby("season")["score_obv_ar1"].count().to_dict()
    cfg.out("indicator_checks.json").write_text(json.dumps(checks, indent=2))
    logger.info(json.dumps(checks))
    return checks


if __name__ == "__main__":
    print(json.dumps(run(load_config()), indent=2))

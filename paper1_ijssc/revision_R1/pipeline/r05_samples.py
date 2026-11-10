"""R05 — Elegibilidad, transiciones y universos (REQ-D8, insumos de REQ-A1, A7, A11).

No calcula ninguna métrica entre temporadas: eso queda para después del preregistro (G2).
Salidas (outputs/r1):
  players_season_full.parquet  players_season + indicador + banderas de elegibilidad
  transitions.parquet          una fila por (jugador con score en t, t, t+k), k = 1..max_k, con estado en t+k
  same_set.parquet             jugadores elegibles en todas las temporadas configuradas
  panel_universe.parquet       base = última temporada, elegibles (D-14)
  samples_summary.json         conteos
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from loguru import logger

from common import Config, load_config

SCORES = ["score_obv_ar1", "score_obv_equal", "score_obv_ar1_macro05", "score_obv_ar1_canonpos", "score_obv_legacy",
          "score_obv_p90"]


def run(cfg: Config) -> dict:
    ps = pd.read_parquet(cfg.out("players_season.parquet"))
    ind = pd.read_parquet(cfg.out("indicator_player_season.parquet"))
    df = ps.merge(ind.drop(columns=["position_group"]), on=["player_id", "season"], how="left")
    vol = cfg.out("indicator_volume.parquet")  # adenda 1: contraste de volumen (no altera el indicador principal)
    if vol.exists():
        df = df.merge(pd.read_parquet(vol)[["player_id", "season", "score_obv_p90"]], on=["player_id", "season"], how="left")
    else:
        df["score_obv_p90"] = float("nan")
    ex = cfg.raw["exposure"]
    has = df.score_obv_ar1.notna() & df.position_group_season.notna()
    df["eligible_main"] = has & (df.minutes_covered >= ex["main_min_minutes"])
    for m in ex["sensitivity_min_minutes"]:
        df[f"eligible_{m}"] = has & (df.minutes_covered >= m)
    df["eligible_share30"] = has & (df.share_club_minutes >= ex["sensitivity_share_club_minutes"])
    df.to_parquet(cfg.out("players_season_full.parquet"), index=False)

    keep = ["player_id", "season", "age_jan1", "position_group_season", "position_group_canonical", "minutes_covered",
            "share_club_minutes", "club_matches_covered", "main_club", "eligible_main", "eligible_500", "eligible_750",
            "eligible_share30"] + SCORES
    base = df[keep]
    rows = []
    for k in range(1, int(cfg.raw["design"]["max_k"]) + 1):
        for t, tk in cfg.transitions(k):
            a = base[(base.season == t) & base.score_obv_ar1.notna()].add_suffix("_t").rename(
                columns={"player_id_t": "player_id"})
            b = base[base.season == tk].add_suffix("_tk").rename(columns={"player_id_tk": "player_id"})
            j = a.merge(b, on="player_id", how="left")
            j["t"], j["tk"], j["k"] = t, tk, k
            j["present_tk"] = j.season_tk.notna()
            j["scored_tk"] = j.score_obv_ar1_tk.notna()
            for c in ["eligible_main", "eligible_500", "eligible_750", "eligible_share30"]:
                j[f"{c}_tk"] = j[f"{c}_tk"].astype("boolean").fillna(False).astype(bool)
            rows.append(j.drop(columns=["season_t", "season_tk"]))
    tr = pd.concat(rows, ignore_index=True)
    tr.to_parquet(cfg.out("transitions.parquet"), index=False)

    el = df[df.eligible_main].pivot_table(index="player_id", columns="season", values="eligible_main", aggfunc="any")
    same = el.dropna().index if el.shape[1] == len(cfg.labels) else pd.Index([])
    pd.DataFrame({"player_id": same}).to_parquet(cfg.out("same_set.parquet"), index=False)

    last = cfg.labels[-1]
    panel = df[(df.season == last) & df.eligible_main][["player_id", "display_name", "season", "age_jan1",
                                                         "position_group_season", "main_club", "minutes_covered",
                                                         "club_matches_covered", "score_obv_ar1"]]
    panel.to_parquet(cfg.out("panel_universe.parquet"), index=False)

    summ = {"eligible_main_by_season_pos": df[df.eligible_main].groupby(["season", "position_group_season"])
            .size().unstack(fill_value=0).to_dict("index"),
            "transitions": {f"{t}->{tk}": {"scored_t": int(len(g)), "eligible_t": int(g.eligible_main_t.sum()),
                                           "eligible_both": int((g.eligible_main_t & g.eligible_main_tk).sum()),
                                           "eligible_t_absent_tk": int((g.eligible_main_t & ~g.present_tk).sum()),
                                           "eligible_t_present_not_eligible_tk": int(
                                               (g.eligible_main_t & g.present_tk & ~g.eligible_main_tk).sum())}
                            for (t, tk), g in tr.groupby(["t", "tk"])},
            "same_set_all_seasons": int(len(same)), "panel_universe": int(len(panel))}
    cfg.out("samples_summary.json").write_text(json.dumps(summ, indent=2, default=str))
    logger.info(json.dumps(summ, default=str))
    return summ


if __name__ == "__main__":
    print(json.dumps(run(load_config()), indent=2, default=str))

"""R03 — Jugadores por temporada: nombre de uso común, nacimiento, edad, posición y club (REQ-D1, D4, D5, D6, D7).

- display_name: `player_nickname` de los lineups (A/B) o `nickname` de los eventos (C); si falta, el nombre.
- birth_date: lineups (A/B) y, si falta, `data/master/players_master.parquet` (incluye Transfermarkt para C).
- age_jan1: edad decimal al 1 de enero de la temporada (D-05).
- position_group_season: grupo modal de los eventos con posición de esa temporada (D-06).
- position_group_canonical: grupo de P01 enviado (solo A/B), para sensibilidad.
- clubs / main_club: desde los minutos de r02 (préstamos y traspasos de mitad de temporada).
Salida: outputs/r1/players_season.parquet (sin el indicador; r04 lo agrega).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from loguru import logger

from common import Config, load_config, pos_group


def run(cfg: Config) -> dict:
    mins = pd.read_parquet(cfg.out("minutes_player_season.parquet"))
    lu = pd.read_parquet(cfg.cache("lineup_player_info.parquet"))
    pmaster = pd.read_parquet(cfg.paths["players_master"])[["player_id", "birth_date"]]
    canon = pd.read_parquet(cfg.paths["submitted_pipeline"] / "p01_canonical_positions.parquet")[
        ["player_id", "position_group"]].rename(columns={"position_group": "position_group_canonical"})
    rows = []
    stats = {}
    for s in cfg.seasons:
        ev = pd.read_parquet(cfg.cache(f"events_{s.label}.parquet"),
                             columns=["player_id", "player_name", "nickname", "position"])
        names = ev.groupby("player_id").agg(player_name=("player_name", "first"),
                                            nick_ev=("nickname", lambda x: x.dropna().iloc[0] if x.notna().any() else None))
        pos = (ev.dropna(subset=["position"]).groupby("player_id")["position"]
               .agg(lambda x: x.value_counts().index[0]).rename("position_detail_season"))
        df = mins[mins.season == s.label].merge(names, on="player_id", how="left").merge(pos, on="player_id", how="left")
        lus = lu[lu.season == s.label].groupby("player_id").agg(
            nick_lu=("nickname", lambda x: x.dropna().iloc[0] if x.notna().any() else None),
            bd_lu=("birth_date", lambda x: x.dropna().iloc[0] if x.notna().any() else None))
        df = df.merge(lus, on="player_id", how="left")
        df["season"] = s.label
        df["jan1"] = pd.Timestamp(s.jan1)
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out["display_name"] = out.nick_lu.fillna(out.nick_ev).fillna(out.player_name)
    # nacimiento: lineup de cualquier temporada del jugador; si no, players_master
    bd_any = out.dropna(subset=["bd_lu"]).groupby("player_id")["bd_lu"].first()
    out["birth_date"] = pd.to_datetime(out.player_id.map(bd_any))
    pm = pmaster.set_index("player_id")["birth_date"]
    miss = out.birth_date.isna()
    out.loc[miss, "birth_date"] = pd.to_datetime(out.loc[miss, "player_id"].map(pm))
    out["birth_source"] = np.where(out.player_id.isin(bd_any.index), "lineup",
                                   np.where(out.birth_date.notna(), "players_master", "missing"))
    out["age_jan1"] = (out.jan1 - out.birth_date).dt.days / 365.25
    out["position_group_season"] = out.position_detail_season.map(pos_group)
    out = out.merge(canon, on="player_id", how="left")
    # nombre de uso común estable por jugador (el más frecuente entre temporadas)
    out["display_name"] = out.groupby("player_id")["display_name"].transform(lambda x: x.value_counts().index[0])
    cols = ["season", "player_id", "player_name", "display_name", "birth_date", "birth_source", "age_jan1",
            "position_detail_season", "position_group_season", "position_group_canonical",
            "minutes_covered", "matches_played", "clubs", "main_club", "club_matches_covered",
            "club_minutes_covered", "clubs_minutes_covered", "share_club_minutes"]
    out = out[cols].sort_values(["season", "player_id"]).reset_index(drop=True)
    out.to_parquet(cfg.out("players_season.parquet"), index=False)
    thr = cfg.raw["exposure"]["main_min_minutes"]
    for s in cfg.labels:
        g = out[out.season == s]
        e = g[g.minutes_covered >= thr]
        stats[s] = {"players": int(len(g)), "eligible_main": int(len(e)),
                    "missing_age_all": int(g.age_jan1.isna().sum()), "missing_age_eligible": int(e.age_jan1.isna().sum()),
                    "missing_position": int(g.position_group_season.isna().sum()),
                    "multi_club": int(g.clubs.str.contains(r"\|").sum()),
                    "birth_source": g.birth_source.value_counts().to_dict()}
    # cambios de grupo entre temporadas consecutivas
    w = out.pivot(index="player_id", columns="season", values="position_group_season")
    ch = {}
    for a, b in cfg.transitions(1):
        both = w[[a, b]].dropna()
        el = out[out.minutes_covered >= thr].pivot(index="player_id", columns="season", values="position_group_season")
        bothe = el[[a, b]].dropna() if a in el and b in el else pd.DataFrame()
        ch[f"{a}->{b}"] = {"in_both": int(len(both)), "changed": int((both[a] != both[b]).sum()),
                           "eligible_both": int(len(bothe)),
                           "eligible_changed": int((bothe[a] != bothe[b]).sum()) if len(bothe) else 0}
    stats["position_changes"] = ch
    ab = out[out.season.isin(["A", "B"])].dropna(subset=["position_group_canonical"])
    stats["season_vs_canonical_differs_AB"] = int((ab.position_group_season != ab.position_group_canonical).sum())
    cfg.out("players_season_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    logger.info(json.dumps(stats, default=str))
    return stats


if __name__ == "__main__":
    print(json.dumps(run(load_config()), indent=2, default=str))

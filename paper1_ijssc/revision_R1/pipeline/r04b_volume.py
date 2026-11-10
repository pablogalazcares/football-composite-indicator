"""R04b — Indicador de volumen para el contraste de la adenda 1 (PREREGISTRO_ADENDA_1 §1).

score_obv_p90 = percentil (0–100, rank promedio) dentro de (temporada, grupo de la temporada), entre los jugadores con
≥ 1.000 minutos reales, de: Σ obv_total_net de todos sus eventos / minutos cubiertos × 90.
Sin parámetros estimados. NaN para jugadores bajo el umbral.
Salida: outputs/r1/indicator_volume.parquet
"""
from __future__ import annotations

import pandas as pd

from common import Config, load_config


def run(cfg: Config) -> dict:
    ps = pd.read_parquet(cfg.out("players_season.parquet"))
    thr = cfg.raw["exposure"]["main_min_minutes"]
    sums = []
    for s in cfg.seasons:
        ev = pd.read_parquet(cfg.cache(f"events_{s.label}.parquet"), columns=["player_id", "obv_total_net"])
        g = ev.groupby("player_id").obv_total_net.sum(min_count=1).rename("obv_total_sum").reset_index()
        sums.append(g.assign(season=s.label))
    v = ps[["player_id", "season", "minutes_covered", "position_group_season"]].merge(
        pd.concat(sums), on=["player_id", "season"], how="left")
    v["obv_p90_raw"] = v.obv_total_sum / v.minutes_covered * 90
    el = (v.minutes_covered >= thr) & v.obv_p90_raw.notna() & v.position_group_season.notna()
    v["score_obv_p90"] = float("nan")
    v.loc[el, "score_obv_p90"] = (v[el].groupby(["season", "position_group_season"]).obv_p90_raw
                                  .rank(method="average", pct=True) * 100)
    v[["player_id", "season", "obv_total_sum", "obv_p90_raw", "score_obv_p90"]].to_parquet(
        cfg.out("indicator_volume.parquet"), index=False)
    return {"n_scored": int(v.score_obv_p90.notna().sum()),
            "by_season": v.groupby("season").score_obv_p90.count().to_dict()}


if __name__ == "__main__":
    print(run(load_config()))

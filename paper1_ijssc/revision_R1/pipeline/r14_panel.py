"""R14 — Panel liberable (D-14; REQ-D8). NO se publica: queda en outputs/r1 hasta que HC-3 confirme la licencia.

Base: última temporada (2024/25), jugadores de campo elegibles. Predicción de M2 (entrenado en A→B) a 1 y 2 temporadas,
con intervalos gaussianos y conformales de 80 % y P(cuartil superior) a 1 temporada. Sin trayectorias deterministas
a 5 años. Solo agregados por jugador-temporada.
Salidas: outputs/r1/panel_R1.parquet, outputs/r1/panel_R1.csv y results/panel_data_dictionary.md
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from scipy.stats import norm

import r06_models as R6
from common import OUTFIELD, Config, load_config


def run(cfg: Config) -> dict:
    c = float(cfg.raw["models"]["age_center"])
    level = float(cfg.raw["metrics"]["interval_level"])
    a, b = cfg.raw["design"]["training_transition"]
    train = R6.pairs(cfg, a, b)
    m2 = R6.Fitted("M2", OUTFIELD, c).fit(train.s_t, train.age_t, train.pos, train.s_tk)
    unc = json.loads(cfg.out("uncertainty.json").read_text())
    sigma, q = unc["sigma_oof"], unc["q_conformal"]
    z = norm.ppf(0.5 + level / 2)
    last = cfg.labels[-1]
    pool = R6.origin_pool(cfg, last)
    ps = pd.read_parquet(cfg.out("players_season_full.parquet"))
    meta = ps[ps.season == last][["player_id", "display_name", "main_club", "minutes_covered", "matches_played",
                                  "club_matches_covered"]]
    out = pool[["player_id", "pos", "age_t", "s_t"]].rename(columns={"pos": "position_group", "age_t": "age_jan1",
                                                                     "s_t": "score_obv"}).merge(meta, on="player_id")
    rng = np.random.default_rng(cfg.seed)
    for k in (1, 2):
        mult = math.sqrt(sum(m2.beta_s ** (2 * j) for j in range(k)))
        pred = m2.project(pool.s_t, pool.age_t, pool.pos, k)
        out[f"pred_k{k}"] = pred
        out[f"lo80_gauss_k{k}"], out[f"hi80_gauss_k{k}"] = pred - z * sigma * mult, pred + z * sigma * mult
        out[f"lo80_conf_k{k}"], out[f"hi80_conf_k{k}"] = pred - q * mult, pred + q * mult
    hits = np.zeros(len(pool))
    posv = pool.pos.values
    mu = out.pred_k1.values
    for _ in range(int(cfg.raw["metrics"]["n_sims_rank"])):
        sim = mu + rng.normal(0, sigma, len(mu))
        for g in OUTFIELD:
            m = posv == g
            hits[m] += sim[m] >= np.quantile(sim[m], 0.75)
    out["p_top_quartile_k1"] = hits / int(cfg.raw["metrics"]["n_sims_rank"])
    out.insert(0, "season_base", cfg.season(last).name)
    out = out.round(3)
    out.to_parquet(cfg.out("panel_R1.parquet"), index=False)
    out.to_csv(cfg.out("panel_R1.csv"), index=False)
    dd = ["# Diccionario del panel R1 (no publicado)", "",
          f"Base: {cfg.season(last).name}; jugadores de campo con ≥ {cfg.raw['exposure']['main_min_minutes']} minutos reales "
          f"en partidos cubiertos. Modelo: M2 (OLS: AR(1) + posición + edad centrada y edad²), entrenado solo en "
          f"{cfg.season(a).name}→{cfg.season(b).name}.", "",
          "| Columna | Descripción |", "|---|---|",
          "| season_base | Temporada de origen (única) |", "| player_id | ID de StatsBomb |",
          "| position_group | Grupo modal de la temporada base |", "| age_jan1 | Edad decimal al 1 de enero de la temporada base |",
          "| score_obv | Indicador OBV dentro de posición (0–100) en la temporada base |",
          "| display_name, main_club | Nombre de uso común y club con más minutos |",
          "| minutes_covered, matches_played, club_matches_covered | Exposición en partidos cubiertos |",
          "| pred_k1, pred_k2 | Predicción puntual a 1 y 2 temporadas |",
          "| lo80_gauss_k*, hi80_gauss_k* | Intervalo gaussiano de 80 % (σ de residuos OOF en la transición de entrenamiento; ×√(1+β²) a k = 2) |",
          "| lo80_conf_k*, hi80_conf_k* | Intervalo conformal de 80 % (cuantil de los residuos absolutos OOF) |",
          "| p_top_quartile_k1 | Probabilidad de estar en el cuartil superior de su posición la temporada siguiente (simulación) |",
          "", "El panel no se deposita hasta confirmar los términos de uso de Hudl para 2024/25 (HC-3)."]
    cfg.res("panel_data_dictionary.md").write_text("\n".join(dd) + "\n", encoding="utf-8")
    return {"n": int(len(out)), "by_pos": out.position_group.value_counts().to_dict()}


if __name__ == "__main__":
    print(run(load_config()))

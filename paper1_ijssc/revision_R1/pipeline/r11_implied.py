"""R11 — Horizonte implícito del modelo frente al observado (PREREGISTRO P6; REQ-A6; D-12).

1. Reproduce el ρk enviado (Spearman de la proyección en k contra k = 0) desde outputs/pipeline/p07, agrupado y por
   posición.
2. Toma los contrafactuales del modelo enviado del diagnóstico d02 (sin edad, edad escalada, βs alternativo), citando
   la procedencia.
3. Para M2 (preregistrado, entrenado en A→B), calcula el ρk implícito del panel con base en la última temporada (k = 1…5)
   y lo compara con la persistencia observada y la precisión a k = 1, 2 (de metrics_main.json).
Escribe outputs/r1/implied.json.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import r06_models as R6
import stats_utils as SU
from common import OUTFIELD, POSITIONS, R1, Config, load_config


def run(cfg: Config) -> dict:
    out: dict = {}
    p07 = pd.read_parquet(cfg.paths["submitted_pipeline"] / "p07_projected_scores.parquet")
    p07["position_group"] = p07.position_group.fillna("MID")
    sub = {"pooled": [SU.spearman(p07[f"score_h{k}"], p07.score_h0) for k in range(1, 6)]}
    for g in POSITIONS:
        m = p07.position_group == g
        sub[g] = [SU.spearman(p07.loc[m, f"score_h{k}"], p07.loc[m, "score_h0"]) for k in range(1, 6)]
    out["submitted_rho_k_reproduced"] = sub
    d02 = json.loads((R1 / "outputs" / "d02_results.json").read_text())
    out["submitted_counterfactuals_from_d02"] = {
        "source": "revision_R1/outputs/d02_results.json (scripts/d02_horizon_mechanics.py)",
        "within_weighted_operative": d02["B_rho_operative"]["within_weighted"],
        "no_age_terms_within": d02["C_counterfactual_rho"]["no_age_terms"]["within_weighted"],
        "age_scale_0.25_pooled": d02["C_counterfactual_rho"]["age_scale_0.25"]["pooled"],
        "age_scale_2.0_pooled": d02["C_counterfactual_rho"]["age_scale_2.0"]["pooled"],
        "beta_0.3_pooled": d02["C_counterfactual_rho"]["beta_0.3"]["pooled"],
        "beta_0.8_pooled": d02["C_counterfactual_rho"]["beta_0.8"]["pooled"],
        "share_variance_from_current_score": d02["C_share_of_projection_variance_from_current_score"],
    }
    # M2 preregistrado: horizonte implícito desde la última temporada
    c = float(cfg.raw["models"]["age_center"])
    a, b = cfg.raw["design"]["training_transition"]
    m2 = R6.Fitted("M2", OUTFIELD, c).fit(*[R6.pairs(cfg, a, b)[col] for col in ("s_t", "age_t", "pos", "s_tk")])
    last = cfg.labels[-1]
    pool = R6.origin_pool(cfg, last)
    imp = []
    for k in range(1, 6):
        pk = m2.project(pool.s_t, pool.age_t, pool.pos, k)
        imp.append(SU.rho_within(pk, pool.s_t, pool.pos, OUTFIELD)[0])
    out["M2_implied_rho_k_from_last_season_W"] = imp
    mm = json.loads(cfg.out("metrics_main.json").read_text())
    obs = {}
    for key, k in (("B->C", 1), ("A->C", 2)):
        v = mm["accuracy"][key]
        obs[f"k{k}"] = {"transition": key, "n": v["n"], "implied_M2_W": v["implied_M2_W"], "observed_persistence_W": v["rho_obs_W"],
                        "accuracy_M2_W": v["rho_M2_W"], "accuracy_M0_W": v["rho_M0_W"]}
    out["implied_vs_observed"] = obs
    out["algebra_note"] = ("Para M0 y M1, la proyección dentro de una posición es una transformación monótona del score "
                           "actual, así que el ρk implícito dentro de posición es 1 en todo k. Solo los términos de edad "
                           "(y posición × edad) lo reducen, sin aportar información observada sobre el futuro.")
    cfg.out("implied.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    print(json.dumps(run(load_config()), indent=1)[:3000])

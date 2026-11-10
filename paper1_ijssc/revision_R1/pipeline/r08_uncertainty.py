"""R08 — Intervalos predictivos, cobertura y P(cuartil superior) (PREREGISTRO P3; D-11).

Intervalos de 80 % para M2, calibrados en A→B:
  gaussiano: ŝ ± z_0.9 · σ, con σ = DE de los residuos OOF (5 folds);
  conformal: ŝ ± q, con q = cuantil ⌈(n+1)·0,8⌉/n de |residuos OOF|.
A k = 2 el ancho se multiplica por √(1 + β̂²).
P(cuartil superior en t+k): simulación gaussiana (2.000 sorteos) sobre los elegibles en t; cuartiles por posición en
cada sorteo sobre esos mismos jugadores; evaluación en los elegibles en ambas temporadas (Brier, confiabilidad).
Escribe outputs/r1/uncertainty.json.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from scipy.stats import norm

import r06_models as R6
import stats_utils as SU
from common import OUTFIELD, Config, load_config


def run(cfg: Config) -> dict:
    c = float(cfg.raw["models"]["age_center"])
    level = float(cfg.raw["metrics"]["interval_level"])
    a, b = cfg.raw["design"]["training_transition"]
    train = R6.pairs(cfg, a, b)
    m2 = R6.Fitted("M2", OUTFIELD, c).fit(train.s_t, train.age_t, train.pos, train.s_tk)
    res = R6.oof_residuals(train, "M2", OUTFIELD, c, cfg.seed)
    sigma = float(np.std(res, ddof=1))
    n = len(res)
    q_conf = float(np.sort(np.abs(res))[min(n - 1, math.ceil((n + 1) * level) - 1)])
    z = float(norm.ppf(0.5 + level / 2))
    out = {"sigma_oof": sigma, "q_conformal": q_conf, "z": z, "beta_s": m2.beta_s, "n_train": n,
           "width_gauss_k1": 2 * z * sigma, "width_conf_k1": 2 * q_conf, "by_transition": {}}
    rng = np.random.default_rng(cfg.seed)
    nsim = int(cfg.raw["metrics"]["n_sims_rank"])
    for k in range(1, int(cfg.raw["design"]["max_k"]) + 1):
        for t, tk in cfg.transitions(k):
            if (t, tk) == (a, b):
                continue
            mult = 1.0 if k == 1 else math.sqrt(sum(m2.beta_s ** (2 * j) for j in range(k)))
            d = R6.pairs(cfg, t, tk)
            pred = m2.project(d.s_t, d.age_t, d.pos, k)
            wg, wc = z * sigma * mult, q_conf * mult
            cov_g = (np.abs(d.s_tk - pred) <= wg)
            cov_c = (np.abs(d.s_tk - pred) <= wc)
            r = {"n": int(len(d)), "multiplier": mult, "coverage_gauss": float(cov_g.mean()),
                 "coverage_conf": float(cov_c.mean()),
                 "coverage_gauss_by_pos": cov_g.groupby(d.pos).mean().to_dict(),
                 "coverage_conf_by_pos": cov_c.groupby(d.pos).mean().to_dict()}
            # bootstrap de la cobertura
            bt = SU.bootstrap(lambda ii: {"g": cov_g.values[ii].mean(), "c": cov_c.values[ii].mean()},
                              d.pos, OUTFIELD, int(cfg.raw["metrics"]["bootstrap_B"]), cfg.seed)
            r["coverage_gauss_ci95"], r["coverage_conf_ci95"] = SU.ci(bt.g), SU.ci(bt.c)
            # P(cuartil superior) sobre los elegibles en t
            pool = R6.origin_pool(cfg, t)
            mu = m2.project(pool.s_t, pool.age_t, pool.pos, k)
            sd = sigma * mult
            hits = np.zeros(len(pool))
            posv = pool.pos.values
            for _ in range(nsim):
                sim = mu + rng.normal(0, sd, len(pool))
                for g in OUTFIELD:
                    m = posv == g
                    thr = np.quantile(sim[m], 0.75)
                    hits[m] += sim[m] >= thr
            pool = pool.assign(p_topq=hits / nsim)
            # cuartil observado en t+k (entre los elegibles en ambas, por posición) y cuartil en t (entre los elegibles en t)
            pool["q_t"] = pool.groupby("pos").s_t.transform(lambda x: pd.qcut(x.rank(method="first"), 4, labels=[1, 2, 3, 4]))
            ev = d[["player_id", "s_tk", "pos"]].merge(pool[["player_id", "p_topq", "q_t"]], on="player_id")
            ev["obs_top"] = ev.groupby("pos").s_tk.transform(lambda x: x >= x.quantile(0.75)).astype(float)
            brier = float(np.mean((ev.p_topq - ev.obs_top) ** 2))
            brier_base = float(np.mean((0.25 - ev.obs_top) ** 2))
            bins = pd.cut(ev.p_topq, [-0.001, 0.1, 0.25, 0.4, 0.6, 1.0])
            rel = ev.groupby(bins, observed=True).agg(n=("obs_top", "size"), p_mean=("p_topq", "mean"),
                                                       obs_rate=("obs_top", "mean")).reset_index()
            rel["bin"] = rel.p_topq.astype(str)
            cond = ev.groupby("q_t", observed=True).agg(n=("obs_top", "size"), obs_rate=("obs_top", "mean"),
                                                         model_p=("p_topq", "mean")).reset_index()
            cond["q_t"] = cond.q_t.astype(int)
            r.update({"brier": brier, "brier_base_rate": brier_base, "brier_skill": 1 - brier / brier_base,
                      "reliability": rel[["bin", "n", "p_mean", "obs_rate"]].to_dict("records"),
                      "top_quartile_given_quartile_t": cond.to_dict("records"),
                      "n_pool": int(len(pool)), "n_evaluated": int(len(ev))})
            out["by_transition"][f"{t}->{tk}"] = r
            if k == 1:
                pool[["player_id", "pos", "p_topq", "q_t"]].to_parquet(cfg.out(f"p_topq_{t}_{tk}.parquet"), index=False)
    cfg.out("uncertainty.json").write_text(json.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    print(json.dumps(run(load_config()), indent=1, default=str)[:4000])

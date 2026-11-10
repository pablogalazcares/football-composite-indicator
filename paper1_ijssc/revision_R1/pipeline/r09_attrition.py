"""R09 — Atrición, IPW y estratos de cobertura (PREREGISTRO P4; REQ-A7, A8).

Escribe outputs/r1/attrition.json y outputs/r1/attrition_table.csv.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import r06_models as R6
import stats_utils as SU
from common import OUTFIELD, Config, load_config


def _status_table(cfg: Config, t: str, tk: str) -> pd.DataFrame:
    tr = pd.read_parquet(cfg.out("transitions.parquet"))
    d = tr[(tr.t == t) & (tr.tk == tk) & tr.eligible_main_t & tr.position_group_season_t.isin(OUTFIELD)].copy()
    d["status"] = np.where(d.eligible_main_tk, "eligible", np.where(d.present_tk, "present_not_eligible", "absent"))
    d["partial_club"] = d.club_matches_covered_t < 30
    return d


def _ipw_weights(d: pd.DataFrame) -> np.ndarray:
    X = np.column_stack([d.score_obv_ar1_t, d.age_jan1_t, d.age_jan1_t ** 2,
                         (d.position_group_season_t == "MID").astype(float), (d.position_group_season_t == "FWD").astype(float),
                         np.log(d.minutes_covered_t), d.partial_club.astype(float)])
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    y = d.eligible_main_tk.astype(int).values
    lr = LogisticRegression(penalty=None, max_iter=2000).fit((X - mu) / sd, y)
    p = lr.predict_proba((X - mu) / sd)[:, 1]
    w = y.mean() / p
    return np.minimum(w, np.quantile(w[y == 1], 0.99)), y


def run(cfg: Config) -> dict:
    out: dict = {"tables": {}, "ipw": {}, "coverage_strata": {}}
    rows = []
    B = int(cfg.raw["metrics"]["bootstrap_B"])
    for k in range(1, int(cfg.raw["design"]["max_k"]) + 1):
        for t, tk in cfg.transitions(k):
            d = _status_table(cfg, t, tk)
            g = d.groupby("status").agg(n=("player_id", "size"), age_mean=("age_jan1_t", "mean"),
                                        age_ge30=("age_jan1_t", lambda x: float((x >= 30).mean())),
                                        score_mean=("score_obv_ar1_t", "mean"), minutes_mean=("minutes_covered_t", "mean"),
                                        partial_club_share=("partial_club", "mean")).reset_index()
            g["transition"] = f"{t}->{tk}"
            rows.append(g)
            out["tables"][f"{t}->{tk}"] = g.drop(columns="transition").to_dict("records")
            # retención por tramo de edad
            bands = pd.cut(d.age_jan1_t, [0, 23, 26, 29, 32, 50], labels=["<=23", "24-26", "27-29", "30-32", "33+"])
            out["tables"][f"{t}->{tk}|retention_by_age"] = d.groupby(bands, observed=True).eligible_main_tk.mean().round(3).to_dict()
            # IPW
            w, y = _ipw_weights(d)
            ev = d[y == 1].assign(w=w[y == 1])
            unweighted, _ = SU.rho_within(ev.score_obv_ar1_t, ev.score_obv_ar1_tk, ev.position_group_season_t, OUTFIELD)
            weighted = SU.weighted_rho_within(ev.score_obv_ar1_t, ev.score_obv_ar1_tk, ev.w, ev.position_group_season_t, OUTFIELD)
            rng = np.random.default_rng(cfg.seed)
            bt = []
            for _ in range(B):
                ii = SU.stratified_indices(d.position_group_season_t.values, OUTFIELD, rng)
                db = d.iloc[ii].reset_index(drop=True)
                try:
                    wb, yb = _ipw_weights(db)
                except ValueError:
                    continue
                eb = db[yb == 1].assign(w=wb[yb == 1])
                bt.append(SU.weighted_rho_within(eb.score_obv_ar1_t, eb.score_obv_ar1_tk, eb.w,
                                                 eb.position_group_season_t, OUTFIELD))
            out["ipw"][f"{t}->{tk}"] = {"n_eligible_t": int(len(d)), "n_eligible_both": int(y.sum()),
                                        "rho_W_unweighted": unweighted, "rho_W_ipw": weighted,
                                        "rho_W_ipw_ci95": SU.ci(pd.Series(bt)),
                                        "weight_max": float(w[y == 1].max())}
            # estratos de cobertura del club en t (solo tiene sentido si t tiene clubes con cobertura parcial)
            if d.partial_club.any():
                for lab, mask in {"partial_20": d.partial_club, "full_38": ~d.partial_club}.items():
                    sub = d[mask & d.eligible_main_tk]
                    if len(sub) < 15:
                        continue
                    pt, _ = SU.rho_within(sub.score_obv_ar1_t, sub.score_obv_ar1_tk, sub.position_group_season_t, OUTFIELD)
                    bt2 = SU.bootstrap(lambda ii: {"r": SU.rho_within(sub.score_obv_ar1_t.values[ii], sub.score_obv_ar1_tk.values[ii],
                                                                      sub.position_group_season_t.values[ii], OUTFIELD)[0]},
                                       sub.position_group_season_t, OUTFIELD, B, cfg.seed)
                    out["coverage_strata"][f"{t}->{tk}|{lab}"] = {"n": int(len(sub)), "rho_W": pt, "ci95": SU.ci(bt2.r)}
    pd.concat(rows).to_csv(cfg.out("attrition_table.csv"), index=False)
    cfg.out("attrition.json").write_text(json.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    r = run(load_config())
    print(json.dumps({"ipw": r["ipw"], "coverage_strata": r["coverage_strata"]}, indent=1, default=str))
    for k, v in r["tables"].items():
        print(k, v)

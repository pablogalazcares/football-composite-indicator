"""R10 — Validez externa (PREREGISTRO P5; REQ-A9; D-13).

(a) ρ_W(s_t, proporción de minutos del club en t+1) entre los elegibles en t presentes en t+1, y Spearman parcial por
    edad y edad² dentro de posición (promedio ponderado por n).
(b) AUC de s_t para la presencia en la liga en t+1 entre los elegibles en t, excluyendo a los jugadores de clubes
    descendidos al final de t.
(c) Valor de mercado (solo A/B): Spearman parcial de s_A con log VM_B y con Δ log VM, ajustando por edad, edad² y
    posición (dummies).
Escribe outputs/r1/external.json.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import stats_utils as SU
from common import OUTFIELD, Config, load_config

RELEGATED = {"A": {"Leicester City", "Leeds United", "Southampton"},
             "B": {"Burnley", "Luton Town", "Sheffield United"}}


def _partial_within(x, y, age, pos, groups) -> float:
    vals, ns = [], []
    for g in groups:
        m = pos == g
        if m.sum() >= 10:
            vals.append(SU.partial_spearman(x[m], y[m], np.column_stack([age[m], age[m] ** 2])))
            ns.append(m.sum())
    return float(np.average(vals, weights=ns))


def run(cfg: Config) -> dict:
    B = int(cfg.raw["metrics"]["bootstrap_B"])
    tr = pd.read_parquet(cfg.out("transitions.parquet"))
    out: dict = {"minutes_next": {}, "presence_auc": {}, "market_value": {}}
    for t, tk in cfg.transitions(1):
        d = tr[(tr.t == t) & (tr.tk == tk) & tr.eligible_main_t & tr.position_group_season_t.isin(OUTFIELD)]
        pres = d[d.present_tk].reset_index(drop=True)
        x, y, a, p = (pres.score_obv_ar1_t.values, pres.share_club_minutes_tk.values, pres.age_jan1_t.values,
                      pres.position_group_season_t.values)

        def st(ii):
            return {"rho": SU.rho_within(x[ii], y[ii], p[ii], OUTFIELD)[0],
                    "partial": _partial_within(x[ii], y[ii], a[ii], p[ii], OUTFIELD)}
        bt = SU.bootstrap(st, p, OUTFIELD, B, cfg.seed)
        pt = st(np.arange(len(pres)))
        out["minutes_next"][f"{t}->{tk}"] = {"n": int(len(pres)), "rho_W": pt["rho"], "rho_W_ci95": SU.ci(bt.rho),
                                             "partial_age_W": pt["partial"], "partial_age_W_ci95": SU.ci(bt.partial)}
        rel = RELEGATED.get(t, set())
        dd = d[~d.main_club_t.isin(rel)].reset_index(drop=True)
        sc, lab = dd.score_obv_ar1_t.values, dd.present_tk.values
        bt2 = SU.bootstrap(lambda ii: {"auc": SU.auc(sc[ii], lab[ii])}, dd.position_group_season_t, OUTFIELD, B, cfg.seed)
        out["presence_auc"][f"{t}->{tk}"] = {"n": int(len(dd)), "n_absent": int((~lab).sum()), "auc": SU.auc(sc, lab),
                                             "auc_ci95": SU.ci(bt2.auc), "excluded_relegated": sorted(rel)}
    # (c) valor de mercado A/B
    mv = pd.read_parquet(cfg.paths["market_value"])
    mvw = mv.pivot_table(index="player_id", columns="season", values="market_value", aggfunc="first")
    ps = pd.read_parquet(cfg.out("players_season_full.parquet"))
    dA = ps[(ps.season == "A") & ps.eligible_main & ps.position_group_season.isin(OUTFIELD)].copy()
    dA["mvA"], dA["mvB"] = dA.player_id.map(mvw.get("A")), dA.player_id.map(mvw.get("B"))
    for lab, col in {"log_mv_B": None, "dlog_mv": "d"}.items():
        sub = dA.dropna(subset=["mvB"] if col is None else ["mvA", "mvB"])
        sub = sub[(sub.mvB > 0) & ((sub.mvA > 0) if col else True)].reset_index(drop=True)
        yv = np.log(sub.mvB.values) if col is None else np.log(sub.mvB.values) - np.log(sub.mvA.values)
        x, a = sub.score_obv_ar1.values, sub.age_jan1.values
        Z = np.column_stack([a, a ** 2, (sub.position_group_season == "MID").astype(float),
                             (sub.position_group_season == "FWD").astype(float)])
        pt = SU.partial_spearman(x, yv, Z)
        bt3 = SU.bootstrap(lambda ii: {"r": SU.partial_spearman(x[ii], yv[ii], Z[ii])}, sub.position_group_season, OUTFIELD,
                           B, cfg.seed)
        out["market_value"][lab] = {"n": int(len(sub)), "partial_spearman": pt, "ci95": SU.ci(bt3.r),
                                    "raw_spearman": SU.spearman(x, yv)}
    cfg.out("external.json").write_text(json.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    print(json.dumps(run(load_config()), indent=1, default=str))

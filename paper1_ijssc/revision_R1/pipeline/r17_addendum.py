"""R17 — Análisis de la adenda 1 (PREREGISTRO_ADENDA_1 §3): volumen frente a eficiencia y valor de mercado ampliado.

E1: persistencia, precisión (M2 − M0) y validez externa (minutos en t+1, permanencia) del indicador de volumen.
E2: valor de mercado (nivel en t+1, cambio t→t+1, nivel en t) para ambos indicadores en A→B, B→C y C→D′.
E3: contrastes pareados volumen − eficiencia con los mismos remuestreos; correlación entre indicadores.
Escribe outputs/r1/addendum1.json.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import r06_models as R6
import r07_metrics as R7
import stats_utils as SU
from common import OUTFIELD, Config, load_config
from r10_external import RELEGATED

EFF, VOL = "score_obv_ar1", "score_obv_p90"


def _boot(cfg, f, pos, B=None):
    return SU.bootstrap(f, pos, OUTFIELD, B or int(cfg.raw["metrics"]["bootstrap_B"]), cfg.seed)


def _est(point, boot, key):
    return {"est": None if pd.isna(point[key]) else float(point[key]), "ci95": SU.ci(boot[key])}


def _partial_within_age_pos(x, y, age, pos):
    Z = np.column_stack([age, age ** 2, (pos == "MID").astype(float), (pos == "FWD").astype(float)])
    return SU.partial_spearman(x, y, Z)


def run(cfg: Config) -> dict:
    out: dict = {"E1": {}, "E2": {}, "E3": {}, "corr_between_indicators": {}}
    c = float(cfg.raw["models"]["age_center"])
    a, b = cfg.raw["design"]["training_transition"]
    tr = pd.read_parquet(cfg.out("transitions.parquet"))
    ps = pd.read_parquet(cfg.out("players_season_full.parquet"))
    mv = pd.read_parquet(cfg.out("mv_panel.parquet"))
    mvw = mv.pivot_table(index="player_id", columns="season", values="mv_eur", aggfunc="first")

    # correlación entre indicadores dentro de temporada
    for s in cfg.labels:
        d = ps[(ps.season == s) & ps.eligible_main & ps.position_group_season.isin(OUTFIELD)].dropna(subset=[EFF, VOL])
        out["corr_between_indicators"][s] = {"n": int(len(d)),
                                             "rho_W": SU.rho_within(d[EFF], d[VOL], d.position_group_season, OUTFIELD)[0]}

    # E1: persistencia y precisión del volumen; E3a: contraste pareado de persistencia
    fits_vol = R6.fit_all(R6.pairs(cfg, a, b, score=VOL), OUTFIELD, c)
    for t, tk, k in (("B", "C", 1), ("A", "C", 2)):
        r = R7.evaluate(cfg, t, tk, k, fits=fits_vol, score=VOL)
        out["E1"][f"{t}->{tk}"] = {kk: r[kk] for kk in ("n", "rho_obs_W", "rho_obs_DEF", "rho_obs_MID", "rho_obs_FWD",
                                                        "rho_M2_W", "d_rho_M2_W")}
        d = tr[(tr.t == t) & (tr.tk == tk) & tr.eligible_main_t & tr.eligible_main_tk &
               tr.position_group_season_t.isin(OUTFIELD)].dropna(subset=[f"{EFF}_t", f"{EFF}_tk", f"{VOL}_t", f"{VOL}_tk"])
        d = d.reset_index(drop=True)
        p = d.position_group_season_t.values

        def st(ii, d=d, p=p):
            e = SU.rho_within(d[f"{EFF}_t"].values[ii], d[f"{EFF}_tk"].values[ii], p[ii], OUTFIELD)[0]
            v = SU.rho_within(d[f"{VOL}_t"].values[ii], d[f"{VOL}_tk"].values[ii], p[ii], OUTFIELD)[0]
            return {"eff": e, "vol": v, "diff": v - e}
        pt, bt = st(np.arange(len(d))), _boot(cfg, st, p)
        out["E3"][f"persistence_{t}->{tk}"] = {"n": int(len(d)), **{kk: _est(pt, bt, kk) for kk in ("eff", "vol", "diff")}}

    # E1/E3: minutos en t+1 y permanencia (A→B, B→C)
    for t, tk in cfg.transitions(1):
        d = tr[(tr.t == t) & (tr.tk == tk) & tr.eligible_main_t & tr.position_group_season_t.isin(OUTFIELD)]
        pres = d[d.present_tk].dropna(subset=[f"{EFF}_t", f"{VOL}_t"]).reset_index(drop=True)
        p, y = pres.position_group_season_t.values, pres.share_club_minutes_tk.values

        def stm(ii, pres=pres, p=p, y=y):
            e = SU.rho_within(pres[f"{EFF}_t"].values[ii], y[ii], p[ii], OUTFIELD)[0]
            v = SU.rho_within(pres[f"{VOL}_t"].values[ii], y[ii], p[ii], OUTFIELD)[0]
            return {"eff": e, "vol": v, "diff": v - e}
        pt, bt = stm(np.arange(len(pres))), _boot(cfg, stm, p)
        out["E3"][f"minutes_next_{t}->{tk}"] = {"n": int(len(pres)), **{kk: _est(pt, bt, kk) for kk in ("eff", "vol", "diff")}}
        dd = d[~d.main_club_t.isin(RELEGATED.get(t, set()))].dropna(subset=[f"{VOL}_t"]).reset_index(drop=True)
        sc, lab = dd[f"{VOL}_t"].values, dd.present_tk.values
        bt2 = _boot(cfg, lambda ii: {"auc": SU.auc(sc[ii], lab[ii])}, dd.position_group_season_t)
        out["E1"][f"presence_auc_vol_{t}->{tk}"] = {"n": int(len(dd)), "auc": SU.auc(sc, lab), "ci95": SU.ci(bt2.auc)}

    # E2/E3: valor de mercado A→B, B→C, C→D′
    labs = cfg.labels + ["D"]
    for i in range(len(labs) - 1):
        t, tk = labs[i], labs[i + 1]
        d = ps[(ps.season == t) & ps.eligible_main & ps.position_group_season.isin(OUTFIELD)].copy()
        d["mv_t"], d["mv_tk"] = d.player_id.map(mvw.get(t)), d.player_id.map(mvw.get(tk))
        d = d.dropna(subset=["mv_t", "mv_tk", EFF, VOL])
        d = d[(d.mv_t > 0) & (d.mv_tk > 0)].reset_index(drop=True)
        p, age = d.position_group_season.values, d.age_jan1.values
        ylev, ychg, ycon = np.log(d.mv_tk.values), np.log(d.mv_tk.values) - np.log(d.mv_t.values), np.log(d.mv_t.values)

        def stv(ii, d=d, p=p, age=age, ylev=ylev, ychg=ychg, ycon=ycon):
            r = {}
            for nm, col in (("eff", EFF), ("vol", VOL)):
                x = d[col].values[ii]
                r[f"{nm}_level_next"] = _partial_within_age_pos(x, ylev[ii], age[ii], p[ii])
                r[f"{nm}_change"] = _partial_within_age_pos(x, ychg[ii], age[ii], p[ii])
                r[f"{nm}_level_now"] = _partial_within_age_pos(x, ycon[ii], age[ii], p[ii])
            r["diff_level_next"] = r["vol_level_next"] - r["eff_level_next"]
            r["diff_change"] = r["vol_change"] - r["eff_change"]
            return r
        pt, bt = stv(np.arange(len(d))), _boot(cfg, stv, p)
        out["E2"][f"{t}->{tk}"] = {"n": int(len(d)), **{kk: _est(pt, bt, kk) for kk in pt}}
    cfg.out("addendum1.json").write_text(json.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    r = run(load_config())
    f = lambda x: f"{x['est']:.3f} [{x['ci95'][0]:.3f},{x['ci95'][1]:.3f}]" if isinstance(x, dict) and x.get("est") is not None else str(x)
    print("CORR", r["corr_between_indicators"])
    for k, v in r["E1"].items():
        print("E1", k, {kk: f(vv) for kk, vv in v.items()} if isinstance(v, dict) else v)
    for k, v in r["E3"].items():
        print("E3", k, {kk: f(vv) for kk, vv in v.items()})
    for k, v in r["E2"].items():
        print("E2", k, {kk: f(vv) for kk, vv in v.items()})

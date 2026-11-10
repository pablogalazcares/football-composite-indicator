"""R19 — Análisis de robustez de la adenda 2 (PREREGISTRO_ADENDA_2 §1, commit 29d63c2), todos exploratorios.

R1 contexto del club en la persistencia (se quedan / cambian; parcial con dummies de club en t y t+k);
R2 valor de mercado con dummies del club en t;
R3 retención del cuartil superior con cuartiles sobre los evaluados (definición preregistrada de P2);
R4 Δρ_W (M2 − M0) ponderado por IPW;
R5 Δρ_W con bootstrap conjunto de entrenamiento y evaluación;
R6 escala de los resúmenes (DE, rango) y MAE y semiancho en DE;
R7 confiabilidad de P(cuartil superior) con todos los bins e IC de Wilson.
Además (R0, sin análisis nuevos: completa la presentación de cantidades ya preespecificadas):
evaluación completa de la contribución (M0–M3, MAE, cuartil superior; adenda 1, E1), diferencia pareada de MAE
(M2 − M0) e IC de la correlación entre resúmenes dentro de cada temporada.
Escribe outputs/r1/robustness.json.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from scipy.stats import norm

import r06_models as R6
import r07_metrics as R7
import stats_utils as SU
from common import OUTFIELD, Config, load_config
from r09_attrition import _ipw_weights, _status_table

EFF, VOL = "score_obv_ar1", "score_obv_p90"
SCORES = {"eff": EFF, "vol": VOL}
EVAL = (("B", "C", 1), ("A", "C", 2))


def _est(point: float, boot) -> dict:
    return {"est": None if pd.isna(point) else float(point), "ci95": SU.ci(pd.Series(boot))}


def _wilson(k: int, n: int, z: float = 1.959964) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return [c - h, c + h]


def _dummies(v: np.ndarray, drop_first: bool = True) -> np.ndarray:
    levels = sorted(pd.unique(v))[1 if drop_first else 0:]
    return np.column_stack([(v == lv).astype(float) for lv in levels]) if levels else np.zeros((len(v), 0))


def _pairs_both(cfg: Config, t: str, tk: str) -> pd.DataFrame:
    tr = pd.read_parquet(cfg.out("transitions.parquet"))
    d = tr[(tr.t == t) & (tr.tk == tk) & tr.eligible_main_t & tr.eligible_main_tk &
           tr.position_group_season_t.isin(OUTFIELD)]
    d = d.dropna(subset=[f"{EFF}_t", f"{EFF}_tk", f"{VOL}_t", f"{VOL}_tk", "main_club_t", "main_club_tk"])
    return d.assign(pos=d.position_group_season_t).reset_index(drop=True)


def club_context(cfg: Config, B: int) -> dict:
    out = {}
    for t, tk, k in EVAL:
        d = _pairs_both(cfg, t, tk)
        stay = (d.main_club_t == d.main_club_tk).values
        res = {"n": int(len(d)), "n_stayers": int(stay.sum()), "n_movers": int((~stay).sum())}
        # (a) se quedan / cambian, con bootstrap independiente por subgrupo
        for nm, col in SCORES.items():
            sub_est, sub_boot = {}, {}
            for lab, m in (("stayers", stay), ("movers", ~stay)):
                s = d[m].reset_index(drop=True)
                x, y, p = s[f"{col}_t"].values, s[f"{col}_tk"].values, s.pos.values
                sub_est[lab] = SU.rho_within(x, y, p, OUTFIELD)[0]
                sub_boot[lab] = SU.bootstrap(lambda ii, x=x, y=y, p=p: {"r": SU.rho_within(x[ii], y[ii], p[ii], OUTFIELD)[0]},
                                             p, OUTFIELD, B, cfg.seed + (0 if lab == "stayers" else 1)).r.values
                res[f"{nm}_{lab}"] = _est(sub_est[lab], sub_boot[lab])
            res[f"{nm}_stay_minus_move"] = _est(sub_est["stayers"] - sub_est["movers"], sub_boot["stayers"] - sub_boot["movers"])
        # (b) parcial con posición (referencia) y con posición + club en t + club en t+k; pareado contribución − eficiencia
        P = _dummies(d.pos.values)
        C = np.column_stack([P, _dummies(d.main_club_t.values), _dummies(d.main_club_tk.values)])

        def st(ii):
            r = {}
            for nm, col in SCORES.items():
                x, y = d[f"{col}_t"].values[ii], d[f"{col}_tk"].values[ii]
                r[f"{nm}_pos"] = SU.partial_spearman(x, y, P[ii])
                r[f"{nm}_club"] = SU.partial_spearman(x, y, C[ii])
            r["diff_pos"], r["diff_club"] = r["vol_pos"] - r["eff_pos"], r["vol_club"] - r["eff_club"]
            return r
        pt, bt = st(np.arange(len(d))), SU.bootstrap(st, d.pos, OUTFIELD, B, cfg.seed)
        res["partial"] = {kk: _est(pt[kk], bt[kk]) for kk in pt}
        out[f"{t}->{tk}"] = res
    return out


def market_value_club(cfg: Config, B: int) -> dict:
    ps = pd.read_parquet(cfg.out("players_season_full.parquet"))
    mv = pd.read_parquet(cfg.out("mv_panel.parquet"))
    mvw = mv.pivot_table(index="player_id", columns="season", values="mv_eur", aggfunc="first")
    labs = cfg.labels + ["D"]
    out = {}
    for i in range(len(labs) - 1):
        t, tk = labs[i], labs[i + 1]
        d = ps[(ps.season == t) & ps.eligible_main & ps.position_group_season.isin(OUTFIELD)].copy()
        d["mv_t"], d["mv_tk"] = d.player_id.map(mvw.get(t)), d.player_id.map(mvw.get(tk))
        d = d.dropna(subset=["mv_t", "mv_tk", EFF, VOL, "main_club"])
        d = d[(d.mv_t > 0) & (d.mv_tk > 0)].reset_index(drop=True)
        age, pos = d.age_jan1.values, d.position_group_season.values
        Zb = np.column_stack([age, age ** 2, _dummies(pos)])
        Zc = np.column_stack([Zb, _dummies(d.main_club.values)])
        ys = {"level_next": np.log(d.mv_tk.values), "level_now": np.log(d.mv_t.values),
              "change": np.log(d.mv_tk.values) - np.log(d.mv_t.values)}

        def st(ii):
            r = {}
            for yn, y in ys.items():
                for nm, col in SCORES.items():
                    x = d[col].values[ii]
                    r[f"{nm}_{yn}_base"] = SU.partial_spearman(x, y[ii], Zb[ii])
                    r[f"{nm}_{yn}_club"] = SU.partial_spearman(x, y[ii], Zc[ii])
                r[f"diff_{yn}_club"] = r[f"vol_{yn}_club"] - r[f"eff_{yn}_club"]
            return r
        pt, bt = st(np.arange(len(d))), SU.bootstrap(st, pos, OUTFIELD, B, cfg.seed)
        out[f"{t}->{tk}"] = {"n": int(len(d)), "n_clubs": int(d.main_club.nunique()),
                             **{kk: _est(pt[kk], bt[kk]) for kk in pt}}
    return out


def top_quartile_eval_set(cfg: Config, unc: dict) -> dict:
    """Cuartiles en t sobre los evaluados (definición de P2); frecuencia observada de estar arriba en t+k."""
    c = float(cfg.raw["models"]["age_center"])
    a, b = cfg.raw["design"]["training_transition"]
    rng = np.random.default_rng(cfg.seed)
    nsim = int(cfg.raw["metrics"]["n_sims_rank"])
    out = {}
    for t, tk, k in EVAL:
        for nm, col in SCORES.items():
            d = R6.pairs(cfg, t, tk, score=col)
            # misma convención que stats_utils.top_quartile_precision: arriba = valor ≥ cuantil 0,75 del grupo
            d["q_t"] = d.groupby("pos").s_t.transform(
                lambda x: 1 + (x >= x.quantile(0.25)).astype(int) + (x >= x.quantile(0.5)).astype(int)
                + (x >= x.quantile(0.75)).astype(int))
            d["top_tk"] = d.groupby("pos").s_tk.transform(lambda x: x >= x.quantile(0.75))
            if nm == "eff":  # probabilidad del modelo M2, simulada sobre los mismos evaluados
                tr_ = R6.pairs(cfg, a, b, score=col)
                m2 = R6.Fitted("M2", OUTFIELD, c).fit(tr_.s_t, tr_.age_t, tr_.pos, tr_.s_tk)
                mu = m2.project(d.s_t, d.age_t, d.pos, k)
                sd = unc["sigma_oof"] * unc["by_transition"][f"{t}->{tk}"]["multiplier"]
                hits, posv = np.zeros(len(d)), d.pos.values
                for _ in range(nsim):
                    sim = mu + rng.normal(0, sd, len(d))
                    for g in OUTFIELD:
                        m = posv == g
                        hits[m] += sim[m] >= np.quantile(sim[m], 0.75)
                d["p_model"] = hits / nsim
            rows = []
            for q in (4, 3, 2, 1):
                s = d[d.q_t == q]
                kk, n = int(s.top_tk.sum()), int(len(s))
                row = {"q_t": q, "n": n, "obs": kk / n, "obs_ci95": _wilson(kk, n)}
                if nm == "eff":
                    row["model"] = float(s.p_model.mean())
                rows.append(row)
            out[f"{nm}|{t}->{tk}"] = rows
    return out


def ipw_delta(cfg: Config, B: int) -> dict:
    c = float(cfg.raw["models"]["age_center"])
    a, b = cfg.raw["design"]["training_transition"]
    out = {}
    for nm, col in SCORES.items():
        train = R6.pairs(cfg, a, b, score=col)
        m2 = R6.Fitted("M2", OUTFIELD, c).fit(train.s_t, train.age_t, train.pos, train.s_tk)
        for t, tk, k in EVAL:
            d = _status_table(cfg, t, tk).dropna(subset=[f"{col}_t"]).reset_index(drop=True)

            def stat(db):
                w, y = _ipw_weights(db)
                ev = db[y == 1].assign(w=w[y == 1]).dropna(subset=[f"{col}_tk"])
                p, s, obs = ev.position_group_season_t.values, ev[f"{col}_t"].values, ev[f"{col}_tk"].values
                pred = m2.project(s, ev.age_jan1_t.values, p, k)
                r0 = SU.weighted_rho_within(s, obs, ev.w.values, p, OUTFIELD)
                r2 = SU.weighted_rho_within(pred, obs, ev.w.values, p, OUTFIELD)
                return r2 - r0
            pt = stat(d)
            rng = np.random.default_rng(cfg.seed)
            bt = []
            for _ in range(B):
                ii = SU.stratified_indices(d.position_group_season_t.values, OUTFIELD, rng)
                try:
                    bt.append(stat(d.iloc[ii].reset_index(drop=True)))
                except ValueError:
                    continue
            out[f"{nm}|{t}->{tk}"] = _est(pt, bt)
    return out


def joint_bootstrap(cfg: Config, B: int) -> dict:
    c = float(cfg.raw["models"]["age_center"])
    a, b = cfg.raw["design"]["training_transition"]
    out = {}
    for nm, col in SCORES.items():
        train = R6.pairs(cfg, a, b, score=col)
        for t, tk, k in EVAL:
            ev = R6.pairs(cfg, t, tk, score=col)
            m2 = R6.Fitted("M2", OUTFIELD, c).fit(train.s_t, train.age_t, train.pos, train.s_tk)

            def delta(f, e):
                r0 = SU.rho_within(e.s_t, e.s_tk, e.pos, OUTFIELD)[0]
                r2 = SU.rho_within(f.project(e.s_t, e.age_t, e.pos, k), e.s_tk, e.pos, OUTFIELD)[0]
                return r2 - r0
            pt = delta(m2, ev)
            rng = np.random.default_rng(cfg.seed)
            bt = []
            for _ in range(B):
                tr_b = train.iloc[SU.stratified_indices(train.pos.values, OUTFIELD, rng)]
                ev_b = ev.iloc[SU.stratified_indices(ev.pos.values, OUTFIELD, rng)]
                f = R6.Fitted("M2", OUTFIELD, c).fit(tr_b.s_t, tr_b.age_t, tr_b.pos, tr_b.s_tk)
                bt.append(delta(f, ev_b))
            out[f"{nm}|{t}->{tk}"] = _est(pt, bt)
    return out


def scale(cfg: Config, unc: dict, mm: dict) -> dict:
    ps = pd.read_parquet(cfg.out("players_season_full.parquet"))
    out = {}
    for s in cfg.labels:
        d = ps[(ps.season == s) & ps.eligible_main & ps.position_group_season.isin(OUTFIELD)]
        for nm, col in SCORES.items():
            x = d[col].dropna()
            out[f"{nm}|{s}"] = {"n": int(len(x)), "sd": float(x.std(ddof=1)), "min": float(x.min()), "max": float(x.max()),
                                "median": float(x.median())}
    # MAE y semiancho del intervalo de la eficiencia, en DE de la temporada evaluada (C)
    sd_c = out["eff|C"]["sd"]
    xc = ps[(ps.season == "C") & ps.eligible_main & ps.position_group_season.isin(OUTFIELD)][EFF].dropna().values
    hw1 = unc["z"] * unc["sigma_oof"]
    med = float(np.median(xc))
    out["eff_interval"] = {"half_width_k1": hw1, "half_width_k1_sd": hw1 / sd_c,
                           "median_player_pct_lo": float((xc < med - hw1).mean()),
                           "median_player_pct_hi": float((xc <= med + hw1).mean())}
    for key in ("B->C", "A->C"):
        a_ = mm["accuracy"][key]
        out[f"eff_mae|{key}"] = {m: a_[f"mae_{m}"]["est"] / sd_c for m in ("M0", "M2")}
    return out


def calibration(unc: dict) -> dict:
    out = {}
    for key, r in unc["by_transition"].items():
        out[key] = [{**b, "obs_ci95": _wilson(int(round(b["obs_rate"] * b["n"])), int(b["n"]))} for b in r["reliability"]]
    return out


def completed_reporting(cfg: Config, B: int) -> dict:
    c = float(cfg.raw["models"]["age_center"])
    a, b = cfg.raw["design"]["training_transition"]
    out = {"vol_accuracy": {}, "mae_diff": {}, "corr_between": {}}
    for nm, col in SCORES.items():
        train = R6.pairs(cfg, a, b, score=col)
        fits = R6.fit_all(train, OUTFIELD, c)
        for t, tk, k in EVAL:
            if nm == "vol":
                out["vol_accuracy"][f"{t}->{tk}"] = R7.evaluate(cfg, t, tk, k, fits=fits, score=col)
            d = R6.pairs(cfg, t, tk, score=col)
            e0 = np.abs(d.s_t.values - d.s_tk.values)
            e2 = np.abs(fits["M2"].project(d.s_t, d.age_t, d.pos, k) - d.s_tk.values)
            bt = SU.bootstrap(lambda ii: {"d": e2[ii].mean() - e0[ii].mean()}, d.pos, OUTFIELD, B, cfg.seed)
            out["mae_diff"][f"{nm}|{t}->{tk}"] = _est(e2.mean() - e0.mean(), bt.d)
    ps = pd.read_parquet(cfg.out("players_season_full.parquet"))
    for s_ in cfg.labels:
        d = ps[(ps.season == s_) & ps.eligible_main & ps.position_group_season.isin(OUTFIELD)].dropna(subset=[EFF, VOL])
        x, y, p = d[EFF].values, d[VOL].values, d.position_group_season.values
        bt = SU.bootstrap(lambda ii: {"r": SU.rho_within(x[ii], y[ii], p[ii], OUTFIELD)[0]}, p, OUTFIELD, B, cfg.seed)
        out["corr_between"][s_] = {"n": int(len(d)), **_est(SU.rho_within(x, y, p, OUTFIELD)[0], bt.r)}
    return out


def run(cfg: Config) -> dict:
    B = int(cfg.raw["metrics"]["bootstrap_B"])
    unc = json.loads(cfg.out("uncertainty.json").read_text())
    mm = json.loads(cfg.out("metrics_main.json").read_text())
    out = {"R0_completed": completed_reporting(cfg, B), "R1_club_context": club_context(cfg, B), "R2_market_value_club": market_value_club(cfg, B),
           "R3_top_quartile": top_quartile_eval_set(cfg, unc), "R4_ipw_delta": ipw_delta(cfg, B),
           "R5_joint_bootstrap": joint_bootstrap(cfg, B), "R6_scale": scale(cfg, unc, mm), "R7_calibration": calibration(unc)}
    cfg.out("robustness.json").write_text(json.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    r = run(load_config())
    f = lambda x: f"{x['est']:.3f} [{x['ci95'][0]:.3f}, {x['ci95'][1]:.3f}]" if isinstance(x, dict) and "est" in x else x
    for sec, v in r.items():
        print("==", sec)
        for k, vv in v.items():
            print(" ", k, {kk: f(x) for kk, x in vv.items()} if isinstance(vv, dict) else vv)

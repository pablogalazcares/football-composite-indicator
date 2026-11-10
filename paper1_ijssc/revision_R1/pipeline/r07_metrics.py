"""R07 — Persistencia observada (P1) y precisión fuera de tiempo (P2), con bootstrap estratificado.

`evaluate()` se reutiliza en r12 (sensibilidades). Escribe outputs/r1/metrics_main.json.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import r06_models as R6
import stats_utils as SU
from common import OUTFIELD, Config, load_config


def _metrics(d: pd.DataFrame, idx: np.ndarray, groups, models) -> dict:
    x = d.iloc[idx]
    out = {}
    out["rho_obs_W"], per = SU.rho_within(x.s_t, x.s_tk, x.pos, groups)
    for g in groups:
        out[f"rho_obs_{g}"] = per[g]
    out["rho_obs_P"] = SU.spearman(x.s_t, x.s_tk)
    for m in models:
        p = x[f"pred_{m}"]
        out[f"rho_{m}_W"], _ = SU.rho_within(p, x.s_tk, x.pos, groups)
        out[f"rho_{m}_P"] = SU.spearman(p, x.s_tk)
        out[f"mae_{m}"] = float(np.mean(np.abs(p - x.s_tk)))
        out[f"topq_{m}"] = SU.top_quartile_precision(p, x.s_tk, x.pos, groups)
        out[f"implied_{m}_W"], _ = SU.rho_within(p, x.s_t, x.pos, groups)
        if m != "M0":
            out[f"d_rho_{m}_W"] = out[f"rho_{m}_W"] - out["rho_M0_W"]
            out[f"d_rho_{m}_P"] = out[f"rho_{m}_P"] - out["rho_M0_P"]
    return out


def evaluate(cfg: Config, t: str, tk: str, k: int, fits: dict | None = None, score: str = "score_obv_ar1",
             elig: str = "eligible_main", groups=OUTFIELD, pos_col: str = "position_group_season",
             same_group: bool = False, restrict_ids=None, extra_filter=None, B: int | None = None) -> dict:
    d = R6.pairs(cfg, t, tk, score, elig, groups, pos_col, same_group)
    if restrict_ids is not None:
        d = d[d.player_id.isin(restrict_ids)].reset_index(drop=True)
    if extra_filter is not None:
        d = d[extra_filter(d)].reset_index(drop=True)
    models = []
    if fits is not None:
        for m, f in fits.items():
            d[f"pred_{m}"] = f.project(d.s_t, d.age_t, d.pos, k)
            models.append(m)
    B = B or int(cfg.raw["metrics"]["bootstrap_B"])
    point = _metrics(d, np.arange(len(d)), groups, models)
    boot = SU.bootstrap(lambda ii: _metrics(d, ii, groups, models), d.pos, groups, B, cfg.seed)
    res = {"t": t, "tk": tk, "k": k, "n": int(len(d)), "n_by_pos": d.pos.value_counts().to_dict()}
    for key, v in point.items():
        res[key] = {"est": None if pd.isna(v) else float(v), "ci95": SU.ci(boot[key])}
    return res


def run(cfg: Config) -> dict:
    c = float(cfg.raw["models"]["age_center"])
    a, b = cfg.raw["design"]["training_transition"]
    fits = R6.fit_all(R6.pairs(cfg, a, b), OUTFIELD, c)
    same = set(pd.read_parquet(cfg.out("same_set.parquet")).player_id)
    out = {"persistence": {}, "accuracy": {}}
    for k in range(1, int(cfg.raw["design"]["max_k"]) + 1):
        for t, tk in cfg.transitions(k):
            key = f"{t}->{tk}"
            out["persistence"][key] = evaluate(cfg, t, tk, k)
            out["persistence"][key + "|same_set"] = evaluate(cfg, t, tk, k, restrict_ids=same)
            if (t, tk) != (a, b):
                out["accuracy"][key] = evaluate(cfg, t, tk, k, fits=fits)
    cfg.out("metrics_main.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    r = run(load_config())

    def f(x):
        return f"{x['est']:.3f} [{x['ci95'][0]:.3f}, {x['ci95'][1]:.3f}]" if x["est"] is not None else "NA"

    for k, v in r["persistence"].items():
        print("PERSIST", k, "n", v["n"], "W", f(v["rho_obs_W"]), "| DEF", f(v["rho_obs_DEF"]), "MID", f(v["rho_obs_MID"]),
              "FWD", f(v["rho_obs_FWD"]), "| P", f(v["rho_obs_P"]))
    for k, v in r["accuracy"].items():
        print("ACC", k, "n", v["n"])
        for m in ("M0", "M1", "M2", "M3"):
            line = f"  {m}: W {f(v[f'rho_{m}_W'])} P {f(v[f'rho_{m}_P'])} MAE {v[f'mae_{m}']['est']:.2f} topq {v[f'topq_{m}']['est']:.3f} implied {f(v[f'implied_{m}_W'])}"
            if m != "M0":
                line += f" dW {f(v[f'd_rho_{m}_W'])}"
            print(line)

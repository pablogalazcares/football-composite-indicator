"""R12 — Sensibilidades (PREREGISTRO P8; REQ-A10, A11).

Para cada variante se reajustan M0–M3 en A→B con la misma variante (indicador, elegibilidad, grupos) y se reportan:
la persistencia observada ρ_W a k = 1 (B→C) y k = 2 (A→C), y Δρ_W(M2 − M0) con IC bootstrap.
La variante legacy (indicador enviado + minutos reales) solo existe en A/B: se reporta A→B (in-sample), para R-c.
Escribe outputs/r1/sensitivity.json.
"""
from __future__ import annotations

import json

import pandas as pd

import r06_models as R6
import r07_metrics as R7
from common import OUTFIELD, POSITIONS, Config, load_config

VARIANTS = {
    "main": {},
    "equal_weights": {"score": "score_obv_equal"},
    "macro_0.5": {"score": "score_obv_ar1_macro05"},
    "canonical_position": {"score": "score_obv_ar1_canonpos", "pos_col": "position_group_canonical"},
    "min_500": {"elig": "eligible_500"},
    "min_750": {"elig": "eligible_750"},
    "share_club_30": {"elig": "eligible_share30"},
    "with_goalkeepers": {"groups": tuple(POSITIONS)},
    "same_group_only": {"same_group": True},
    "exposure_1000_1999_at_t": {"extra_filter": lambda d: d.minutes_covered_t < 2000},
    "exposure_ge2000_at_t": {"extra_filter": lambda d: d.minutes_covered_t >= 2000},
}


def _short(r: dict) -> dict:
    keys = ["rho_obs_W", "rho_M0_W", "rho_M2_W", "d_rho_M2_W", "d_rho_M3_W", "mae_M0", "mae_M2"]
    return {"n": r["n"], **{k: r[k] for k in keys if k in r}}


def run(cfg: Config) -> dict:
    c = float(cfg.raw["models"]["age_center"])
    a, b = cfg.raw["design"]["training_transition"]
    out: dict = {}
    for name, v in VARIANTS.items():
        kw = {k: v[k] for k in ("score", "elig", "groups", "pos_col", "same_group") if k in v}
        groups = kw.get("groups", OUTFIELD)
        train = R6.pairs(cfg, a, b, **kw)
        if "extra_filter" in v:
            train = train[v["extra_filter"](train)].reset_index(drop=True)
        fits = R6.fit_all(train, groups, c)
        res = {"n_train": int(len(train))}
        for t, tk, k in (("B", "C", 1), ("A", "C", 2)):
            r = R7.evaluate(cfg, t, tk, k, fits=fits, extra_filter=v.get("extra_filter"), **kw)
            res[f"{t}->{tk}"] = _short(r)
        out[name] = res
    # legacy (indicador enviado con minutos reales): A→B, persistencia in-sample
    leg = R7.evaluate(cfg, a, b, 1, score="score_obv_legacy", groups=OUTFIELD, pos_col="position_group_canonical")
    rev = R7.evaluate(cfg, a, b, 1)
    canon_bc = R7.evaluate(cfg, "B", "C", 1, score="score_obv_ar1_canonpos", pos_col="position_group_canonical")
    out["legacy_vs_revision_AB"] = {"legacy_rho_W": leg["rho_obs_W"], "revision_rho_W": rev["rho_obs_W"],
                                    "n_legacy": leg["n"], "n_revision": rev["n"],
                                    "canonical_proxy_BC_rho_W": canon_bc["rho_obs_W"]}
    cfg.out("sensitivity.json").write_text(json.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    r = run(load_config())
    f = lambda x: f"{x['est']:.3f} [{x['ci95'][0]:.3f},{x['ci95'][1]:.3f}]" if x["est"] is not None else "NA"
    for k, v in r.items():
        if k == "legacy_vs_revision_AB":
            print(k, {kk: (f(vv) if isinstance(vv, dict) else vv) for kk, vv in v.items()})
            continue
        print(k, "ntrain", v["n_train"], "| k1 n", v["B->C"]["n"], "obs", f(v["B->C"]["rho_obs_W"]), "dM2", f(v["B->C"]["d_rho_M2_W"]),
              "| k2 n", v["A->C"]["n"], "obs", f(v["A->C"]["rho_obs_W"]), "dM2", f(v["A->C"]["d_rho_M2_W"]))

"""R18 — Bloque descriptivo de arqueros (D-03): persistencia de ambos indicadores, con IC.

No es parte de las métricas principales (los arqueros están fuera, D-03): solo se describe en el suplemento.
Escribe outputs/r1/goalkeepers.json.
"""
from __future__ import annotations

import json

import r07_metrics as R7
from common import Config, load_config


def run(cfg: Config) -> dict:
    out = {}
    for score in ("score_obv_ar1", "score_obv_p90"):
        for t, tk, k in (("A", "B", 1), ("B", "C", 1), ("A", "C", 2)):
            r = R7.evaluate(cfg, t, tk, k, score=score, groups=("GK",))
            out[f"{score}|{t}->{tk}"] = {"n": r["n"], "rho": r["rho_obs_GK"]}
    cfg.out("goalkeepers.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    print(json.dumps(run(load_config()), indent=1))

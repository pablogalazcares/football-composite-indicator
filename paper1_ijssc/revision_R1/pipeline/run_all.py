"""Orquestador del pipeline R1 (REQ-R1).

Uso:
  .venv/bin/python paper1_ijssc/revision_R1/pipeline/run_all.py [--config <yaml>] [--outputs <dir>] [--results <dir>]
Con --outputs/--results distintos se hace la corrida limpia de verificación; la caché de eventos (D-17) se regenera.
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config  # noqa: E402

STAGES = ["r01_events", "r02_minutes", "r03_players", "r04_indicator", "r04b_volume", "r05_samples", "r05b_data_audit", "r06_models",
          "r07_metrics", "r08_uncertainty", "r09_attrition", "r10_external", "r11_implied", "r12_sensitivity",
          "r15_tm_link", "r17_addendum", "r18_goalkeepers", "r19_robustness", "r13a_evidence_submitted", "r13_report",
          "r14_panel"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--outputs")
    ap.add_argument("--results")
    ap.add_argument("--from-stage", default=STAGES[0])
    args = ap.parse_args()
    ov = {}
    if args.outputs:
        ov["paths.outputs"] = args.outputs
    if args.results:
        ov["paths.results"] = args.results
    cfg = load_config(args.config, ov)
    import numpy, pandas, scipy, sklearn, matplotlib  # noqa: E401
    env = (f"python {platform.python_version()}\npandas {pandas.__version__}\nnumpy {numpy.__version__}\n"
           f"scipy {scipy.__version__}\nscikit-learn {sklearn.__version__}\nmatplotlib {matplotlib.__version__}\n"
           f"seed {cfg.seed}\n")
    cfg.res("environment.txt").write_text(env)
    start = STAGES.index(args.from_stage)
    for st in STAGES[start:]:
        t0 = time.time()
        import_module(st).run(cfg)
        print(f"[run_all] {st} ok ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()

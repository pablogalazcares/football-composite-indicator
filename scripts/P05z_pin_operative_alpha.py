"""
P05z — Pin the operative blend weight to alpha=0.06
===================================================

Rationale
---------
The fold-robust OOF optimum sits at alpha ~= 0.06 (argmax of the mean OOF
rho under 5-fold and 10-fold, and the per-seed median/mode); the previous
alpha=0.10 was the argmax at a single canonical fold partition (seed=42)
and does not survive fold-seed resampling. The predictive advantage over
pure OBV is not statistically significant (bootstrap p=0.76), so the blend
weight is a design choice; we adopt the fold-robust value 0.06.

This re-selects the SAME operative architecture (C_ridge_inter_10.0,
min_minutes=1000) evaluated at alpha=0.06, refits it on all training pairs,
and overwrites the P05 operative artifacts consumed downstream by P06/P07:
  outputs/pipeline/p05_selected_config.csv
  outputs/pipeline/p05_v11_coefficients.csv
  outputs/pipeline/p05_selected_model.pkl

The previous alpha=0.10 artifacts are backed up under
  outputs/pipeline/_backup_alpha010/
so paper 2 (which still consumes alpha=0.10 projections) can be restored /
re-propagated later.

Reuses P05_model_battery functions verbatim (zero methodological drift).
"""

from __future__ import annotations

import pickle
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "pipeline"))

import P05_model_battery as P  # noqa: E402

OUT = REPO / "outputs" / "pipeline"
BACKUP = OUT / "_backup_alpha010"
OPERATIVE_MODEL_ID = "C_ridge_inter_10.0"
MIN_MINUTES = 1000
ALPHA_NEW = 0.06

ARTIFACTS = [
    "p05_selected_config.csv",
    "p05_v11_coefficients.csv",
    "p05_selected_model.pkl",
]


def main() -> None:
    # 1) Backup current (alpha=0.10) operative artifacts
    BACKUP.mkdir(parents=True, exist_ok=True)
    for a in ARTIFACTS:
        src = OUT / a
        if src.exists() and not (BACKUP / a).exists():
            shutil.copy2(src, BACKUP / a)
            print(f"  backup -> {BACKUP / a}")

    # 2) Evaluate the operative architecture at alpha=0.06 (same battery fn)
    print(f"\n  Evaluando batería a α={ALPHA_NEW}, min={MIN_MINUTES}...")
    res = P.run_battery(alpha=ALPHA_NEW, min_minutes=MIN_MINUTES)
    row = res[res["model_id"] == OPERATIVE_MODEL_ID]
    if len(row) != 1:
        raise SystemExit(f"[ERROR] esperaba 1 fila para {OPERATIVE_MODEL_ID}, hay {len(row)}")
    best_row = row.iloc[0]

    print(f"  model_id     = {best_row['model_id']}")
    print(f"  alpha        = {best_row['alpha']}")
    print(f"  min_minutes  = {best_row['min_minutes']}")
    print(f"  qualifies    = {best_row['qualifies']}")
    print(f"  rho_oof(s42) = {best_row['rho_oof']:.4f}")
    for pos in ["GK", "DEF", "MID", "FWD"]:
        print(f"  peak_{pos}      = {best_row.get(f'peak_age_{pos}', '?')}")

    if not bool(best_row["qualifies"]):
        raise SystemExit("[ERROR] la config a α=0.06 NO califica biológicamente — abortar.")

    # 3) Refit final model on all pairs at alpha=0.06 and validate
    final_model, coef_df = P.fit_final_model(best_row)
    P.validate_selected_model(final_model, coef_df, best_row)

    # 4) Overwrite operative artifacts (same format as P05.main)
    with open(OUT / "p05_selected_model.pkl", "wb") as f:
        pickle.dump({"model": final_model, "config": best_row.to_dict()}, f)
    coef_df.to_csv(OUT / "p05_v11_coefficients.csv", index=False)
    pd.DataFrame([best_row.to_dict()]).to_csv(OUT / "p05_selected_config.csv", index=False)

    beta_s = coef_df.loc[coef_df["feature"] == "score_t0", "coefficient"].iloc[0]
    print(f"\n  [OK] Operative alpha pinned to {ALPHA_NEW}. beta_s={beta_s:.4f}")
    print(f"  Overwritten: {[str(OUT / a) for a in ARTIFACTS]}")


if __name__ == "__main__":
    main()

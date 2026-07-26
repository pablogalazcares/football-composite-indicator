"""
P05d — Fine alpha-grid OOF sweep under fold-seed resampling
===========================================================

Aim
---
The headline model-selection number (rho_OOF at alpha*=0.10) is computed on
a SINGLE canonical fold partition (seed=42). P05c showed that the alpha=0
vs alpha=0.1 comparison is not robust to the fold partition. This script
answers the natural follow-up: *where does the OOF maximum actually sit
once we average over fold partitions, and how does it move across the full
alpha in [0,1] grid?*

Method (identical architecture to P05c / outputs/pipeline/p05_v11)
-----------------------------------------------------------------
- Same composite target score = (1-alpha)*OBV + alpha*PSR at t0 and t1.
- Same design matrix (position x age, position x age^2 interactions).
- Same Ridge(lambda=10), same 5-fold OOF Spearman rho.
- alpha grid: 0.00 .. 1.00 in steps of 0.01 (101 values).
- For each alpha, OOF rho is computed under N_FOLD_SEEDS=50 fold seeds
  (1000..1049), matching P05c, plus the canonical seed=42 separately.

The reusable pieces (design_matrix, oof_predictions, constants) are IMPORTED
from P05c_significance to guarantee zero methodological drift.

Outputs
-------
outputs/paper1_smf/p05d_alpha_grid_sweep_long.csv    (alpha, seed, rho)
outputs/paper1_smf/p05d_alpha_grid_summary.csv       (per-alpha robust stats)
outputs/paper1_smf/p05d_alpha_grid_canonical.csv     (alpha, rho at seed=42)

Console prints the robust argmax (max mean-rho alpha) and the canonical
argmax, so the framing decision can be made from evidence.

Notes
-----
- Reads only outputs/pipeline/p04_training_pairs_1000min.parquet.
- Does not modify scripts/pipeline or any manuscript file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

# Import canonical methodology (guarded main => safe to import)
from P05c_significance import (
    TRAINING_PAIRS,
    OUT_DIR,
    N_FOLDS,
    N_FOLD_SEEDS,
    FOLD_SEED_OFFSET,
    SEED_CANONICAL,
    oof_predictions,
)

ALPHA_GRID = np.round(np.arange(0.0, 1.0 + 1e-9, 0.01), 2)


def oof_rho(df: pd.DataFrame, alpha: float, seed: int) -> float:
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    preds, y = oof_predictions(df, alpha=alpha, kf=kf)
    return float(spearmanr(preds, y).statistic)


def main() -> None:
    df = pd.read_parquet(TRAINING_PAIRS)
    print(f"Loaded {len(df)} training pairs from {TRAINING_PAIRS.name}")
    print(f"Alpha grid: {len(ALPHA_GRID)} values, step 0.01, [0, 1]")
    print(f"Fold seeds: {N_FOLD_SEEDS} ({FOLD_SEED_OFFSET}..{FOLD_SEED_OFFSET + N_FOLD_SEEDS - 1}) "
          f"+ canonical seed {SEED_CANONICAL}")

    seeds = [FOLD_SEED_OFFSET + i for i in range(N_FOLD_SEEDS)]

    long_rows = []
    for a in ALPHA_GRID:
        for s in seeds:
            long_rows.append({"alpha": a, "seed": s, "rho": oof_rho(df, a, s)})
    long_df = pd.DataFrame(long_rows)

    # Robust per-alpha summary across the 50 fold seeds
    summary = (
        long_df.groupby("alpha")["rho"]
        .agg(
            mean_rho="mean",
            std_rho="std",
            min_rho="min",
            max_rho="max",
            q025=lambda x: np.percentile(x, 2.5),
            q975=lambda x: np.percentile(x, 97.5),
        )
        .reset_index()
    )
    summary["n_seeds"] = N_FOLD_SEEDS

    # Canonical single-partition curve (seed=42)
    canon = pd.DataFrame(
        {"alpha": ALPHA_GRID,
         "rho_canonical": [oof_rho(df, a, SEED_CANONICAL) for a in ALPHA_GRID]}
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(OUT_DIR / "p05d_alpha_grid_sweep_long.csv", index=False)
    summary.to_csv(OUT_DIR / "p05d_alpha_grid_summary.csv", index=False)
    canon.to_csv(OUT_DIR / "p05d_alpha_grid_canonical.csv", index=False)

    # ----- Evidence for the framing decision -----
    robust_idx = summary["mean_rho"].idxmax()
    a_robust = summary.loc[robust_idx, "alpha"]
    rho_robust = summary.loc[robust_idx, "mean_rho"]

    canon_idx = canon["rho_canonical"].idxmax()
    a_canon = canon.loc[canon_idx, "alpha"]
    rho_canon = canon.loc[canon_idx, "rho_canonical"]

    r_a0 = summary.loc[summary["alpha"] == 0.0, "mean_rho"].iloc[0]
    r_a10 = summary.loc[np.isclose(summary["alpha"], 0.10), "mean_rho"].iloc[0]

    print("\n===== ROBUST (mean over 50 fold seeds) =====")
    print(f"  argmax mean-rho : alpha* = {a_robust:.2f}  (mean rho = {rho_robust:.4f})")
    print(f"  mean rho @ alpha=0.00 : {r_a0:.4f}")
    print(f"  mean rho @ alpha=0.10 : {r_a10:.4f}")
    print(f"  robust Delta(0.10 - 0.00) = {r_a10 - r_a0:+.4f}")

    print("\n===== CANONICAL (single seed = 42) =====")
    print(f"  argmax rho : alpha* = {a_canon:.2f}  (rho = {rho_canon:.4f})")

    # How wide is the near-optimal plateau (within 1 sd of the robust max)?
    sd_at_max = summary.loc[robust_idx, "std_rho"]
    within = summary[summary["mean_rho"] >= rho_robust - sd_at_max]
    print("\n===== PLATEAU (mean-rho within 1 sd of robust max) =====")
    print(f"  alpha in [{within['alpha'].min():.2f}, {within['alpha'].max():.2f}]  "
          f"(sd at max = {sd_at_max:.4f})")

    print("\nWrote:")
    print(f"  {OUT_DIR / 'p05d_alpha_grid_sweep_long.csv'}")
    print(f"  {OUT_DIR / 'p05d_alpha_grid_summary.csv'}")
    print(f"  {OUT_DIR / 'p05d_alpha_grid_canonical.csv'}")


if __name__ == "__main__":
    main()

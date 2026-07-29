"""
P05c06 — Δρ inference at the operative weight alpha*=0.06
========================================================

Recomputes, at the NEW operative blend alpha*=0.06, the same inference the
manuscript reports for the (former) alpha*=0.10:
  - fold-seed sweep (50 seeds) of rho_OOF at alpha=0 and alpha=0.06
    -> refreshes outputs/alpha_selection/p05c_fold_seed_sweep.csv (was stale)
  - canonical-seed rho_OOF at alpha in {0, 0.06, 1}
  - bootstrap (B=5000, players) of Delta rho vs pure OBV and vs pure PSR
  - Williams dependent-correlation test for each comparison

Self-target convention (each alpha predicts its own composite next-season),
identical to P05c_significance. Reuses its functions verbatim.

Outputs
-------
outputs/alpha_selection/p05c_fold_seed_sweep.csv           (refreshed, alpha=0.06)
outputs/alpha_selection/p05c06_significance_summary.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

from P05c_significance import (
    TRAINING_PAIRS, OUT_DIR, N_FOLDS, N_BOOTSTRAP,
    SEED_CANONICAL, N_FOLD_SEEDS, FOLD_SEED_OFFSET,
    oof_predictions, williams_test,
)

ALPHA_STAR = 0.06


def rho(df, a, kf):
    p, y = oof_predictions(df, alpha=a, kf=kf)
    return float(spearmanr(p, y).statistic), p, y


def main():
    df = pd.read_parquet(TRAINING_PAIRS)
    n = len(df)

    # ---- fold-seed sweep (refresh stale CSV) ----
    seeds = np.arange(N_FOLD_SEEDS) + FOLD_SEED_OFFSET
    r0 = np.empty(N_FOLD_SEEDS); rS = np.empty(N_FOLD_SEEDS)
    for i, s in enumerate(seeds):
        kf = KFold(N_FOLDS, shuffle=True, random_state=int(s))
        r0[i] = rho(df, 0.0, kf)[0]
        rS[i] = rho(df, ALPHA_STAR, kf)[0]
    dsweep = rS - r0
    pd.DataFrame({"seed": seeds, "rho_alpha_0": r0,
                  f"rho_alpha_{ALPHA_STAR}": rS, "delta_rho": dsweep}
                 ).to_csv(OUT_DIR / "p05c_fold_seed_sweep.csv", index=False)

    # ---- canonical seed ----
    kf = KFold(N_FOLDS, shuffle=True, random_state=SEED_CANONICAL)
    rc0, p0, y = rho(df, 0.0, kf)
    rcS, pS, _ = rho(df, ALPHA_STAR, kf)
    rc1, p1, _ = rho(df, 1.0, kf)
    r_S0 = spearmanr(pS, p0).statistic
    r_S1 = spearmanr(pS, p1).statistic
    w_t0, w_p0 = williams_test(rcS, rc0, r_S0, n)
    w_t1, w_p1 = williams_test(rcS, rc1, r_S1, n)

    # ---- bootstrap over players (canonical fold seed) ----
    rng = np.random.default_rng(SEED_CANONICAL)
    idx = np.arange(n)
    d0 = np.empty(N_BOOTSTRAP); d1 = np.empty(N_BOOTSTRAP)
    rS_b = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        bi = rng.choice(idx, size=n, replace=True)
        dfb = df.iloc[bi].reset_index(drop=True)
        kfb = KFold(N_FOLDS, shuffle=True, random_state=SEED_CANONICAL)
        rb0 = rho(dfb, 0.0, kfb)[0]
        rbS = rho(dfb, ALPHA_STAR, kfb)[0]
        rb1 = rho(dfb, 1.0, kfb)[0]
        d0[b] = rbS - rb0; d1[b] = rbS - rb1; rS_b[b] = rbS

    def ci(x): return np.percentile(x, [2.5, 97.5])
    def p2(x): return 2 * min((x <= 0).mean(), (x >= 0).mean())

    rows = [
        {"quantity": "rho_oof_alpha_0_canonical", "value": rc0},
        {"quantity": "rho_oof_alpha_star_canonical", "value": rcS},
        {"quantity": "rho_oof_alpha_1_canonical", "value": rc1},
        {"quantity": "rho_oof_alpha_star_boot_mean", "value": rS_b.mean(),
         "ci_lo": ci(rS_b)[0], "ci_hi": ci(rS_b)[1]},
        {"quantity": "delta_rho0_vs_OBV", "value": d0.mean(),
         "ci_lo": ci(d0)[0], "ci_hi": ci(d0)[1], "p_boot": p2(d0),
         "williams_t": w_t0, "williams_p": w_p0},
        {"quantity": "delta_rho1_vs_PSR", "value": d1.mean(),
         "ci_lo": ci(d1)[0], "ci_hi": ci(d1)[1], "p_boot": p2(d1),
         "williams_t": w_t1, "williams_p": w_p1},
        {"quantity": "fold_seed_mean_rho0", "value": r0.mean(),
         "ci_lo": r0.min(), "ci_hi": r0.max()},
        {"quantity": "fold_seed_mean_rhoStar", "value": rS.mean(),
         "ci_lo": rS.min(), "ci_hi": rS.max()},
        {"quantity": "fold_seed_mean_delta", "value": dsweep.mean(),
         "ci_lo": np.percentile(dsweep, 2.5), "ci_hi": np.percentile(dsweep, 97.5)},
        {"quantity": "fold_seed_pct_delta_pos", "value": float((dsweep > 0).mean())},
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "p05c06_significance_summary.csv", index=False)
    print(out.to_string(index=False))
    print(f"\n[OK] alpha*={ALPHA_STAR}: canonical rho0={rc0:.4f} rhoStar={rcS:.4f} rho1={rc1:.4f}")
    print(f"  Δρ0 vs OBV = {d0.mean():+.4f} CI[{ci(d0)[0]:+.4f},{ci(d0)[1]:+.4f}] p={p2(d0):.3f} | Williams p={w_p0:.3f}")
    print(f"  Δρ1 vs PSR = {d1.mean():+.4f} CI[{ci(d1)[0]:+.4f},{ci(d1)[1]:+.4f}] p={p2(d1):.3f} | Williams p={w_p1:.3f}")
    print(f"  fold-seed: rho0={r0.mean():.4f} rhoStar={rS.mean():.4f} Δ={dsweep.mean():+.4f} %pos={(dsweep>0).mean()*100:.0f}%")


if __name__ == "__main__":
    main()

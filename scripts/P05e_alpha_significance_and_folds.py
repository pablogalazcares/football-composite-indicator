"""
P05e — Is the OOF gain of a small PSR weight over pure OBV significant?
      + fold-scheme robustness (Paso 4: 5-fold / 10-fold / LOO)
=======================================================================

Decision question
-----------------
The per-seed argmax of OOF rho clusters at alpha ~= 0.06 (median), above
pure OBV (alpha=0). Before adopting alpha ~= 0.05-0.06 as the operative
weight, we must check whether its OOF advantage over pure OBV is
STATISTICALLY SIGNIFICANT — not just the maximum of a noisy curve
(winner's curse: alpha* is selected on the same data used to score it).

Tests
-----
A) Per-seed PAIRED delta rho(alpha) - rho(0) across 50 fold seeds, for
   alpha in {0.05, 0.06, 0.10}: mean, % seeds > 0, paired t-test,
   Wilcoxon signed-rank, 2.5-97.5 percentile band. (5-fold and 10-fold)
B) BOOTSTRAP over players at the canonical seed=42 for alpha in
   {0.05, 0.06} vs 0: 95% CI of delta rho and two-sided p.
C) Williams' dependent-correlation test at canonical seed, alpha=0.06 vs 0.
D) Fold-scheme robustness: robust argmax of mean OOF rho under 5-fold,
   10-fold (50 seeds) and LOO (deterministic).

Reuses design_matrix / oof_predictions from P05c (zero drift).

Outputs
-------
outputs/paper1_smf/p05e_significance_summary.csv
outputs/paper1_smf/p05e_fold_scheme_argmax.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_rel, wilcoxon
from sklearn.model_selection import KFold, LeaveOneOut

from P05c_significance import (
    TRAINING_PAIRS, OUT_DIR, SEED_CANONICAL, FOLD_SEED_OFFSET,
    oof_predictions, williams_test,
)

ALPHA_GRID = np.round(np.arange(0.0, 1.0 + 1e-9, 0.01), 2)
N_SEEDS = 50
B_BOOT = 2000
TEST_ALPHAS = [0.05, 0.06, 0.10]


def rho_at(df, alpha, kf):
    p, y = oof_predictions(df, alpha=alpha, kf=kf)
    return float(spearmanr(p, y).statistic)


def per_seed_delta(df, n_splits):
    seeds = [FOLD_SEED_OFFSET + i for i in range(N_SEEDS)]
    base = np.array([rho_at(df, 0.0, KFold(n_splits, shuffle=True, random_state=s)) for s in seeds])
    out = {}
    for a in TEST_ALPHAS:
        ra = np.array([rho_at(df, a, KFold(n_splits, shuffle=True, random_state=s)) for s in seeds])
        out[a] = ra - base
    return base, out


def main():
    df = pd.read_parquet(TRAINING_PAIRS)
    n = len(df)
    rows = []

    # ---- A) per-seed paired delta, 5-fold and 10-fold ----
    for n_splits in (5, 10):
        _, deltas = per_seed_delta(df, n_splits)
        for a, d in deltas.items():
            t_stat, t_p = ttest_rel(d + 0, np.zeros_like(d))  # H0: mean delta = 0
            try:
                w_stat, w_p = wilcoxon(d)
            except ValueError:
                w_p = np.nan
            rows.append({
                "test": f"per_seed_paired_{n_splits}fold",
                "alpha": a, "mean_delta": d.mean(),
                "pct_seeds_pos": 100 * (d > 0).mean(),
                "q2.5": np.percentile(d, 2.5), "q97.5": np.percentile(d, 97.5),
                "paired_t_p": t_p, "wilcoxon_p": w_p,
            })

    # ---- B) bootstrap over players at canonical seed ----
    rng = np.random.default_rng(SEED_CANONICAL)
    idx_all = np.arange(n)
    for a in (0.05, 0.06):
        deltas_b = np.empty(B_BOOT)
        for b in range(B_BOOT):
            idx = rng.choice(idx_all, size=n, replace=True)
            db = df.iloc[idx].reset_index(drop=True)
            kf = KFold(5, shuffle=True, random_state=SEED_CANONICAL)
            deltas_b[b] = rho_at(db, a, kf) - rho_at(db, 0.0, kf)
        lo, hi = np.percentile(deltas_b, [2.5, 97.5])
        p_two = 2 * min((deltas_b <= 0).mean(), (deltas_b >= 0).mean())
        rows.append({
            "test": "bootstrap_players_canonical", "alpha": a,
            "mean_delta": deltas_b.mean(), "pct_seeds_pos": 100 * (deltas_b > 0).mean(),
            "q2.5": lo, "q97.5": hi, "paired_t_p": p_two, "wilcoxon_p": np.nan,
        })

    # ---- C) Williams at canonical seed, alpha=0.06 vs 0 ----
    kf = KFold(5, shuffle=True, random_state=SEED_CANONICAL)
    p0, y = oof_predictions(df, 0.0, kf)
    p6, _ = oof_predictions(df, 0.06, kf)
    r_y0 = spearmanr(p0, y).statistic
    r_y6 = spearmanr(p6, y).statistic
    r_06 = spearmanr(p0, p6).statistic
    w_t, w_p = williams_test(r_y6, r_y0, r_06, n)
    rows.append({
        "test": "williams_canonical_0.06_vs_0", "alpha": 0.06,
        "mean_delta": r_y6 - r_y0, "pct_seeds_pos": np.nan,
        "q2.5": np.nan, "q97.5": np.nan, "paired_t_p": w_p, "wilcoxon_p": np.nan,
    })

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "p05e_significance_summary.csv", index=False)

    # ---- D) fold-scheme argmax ----
    fold_rows = []
    # 5-fold and 10-fold: mean over seeds
    for n_splits in (5, 10):
        seeds = [FOLD_SEED_OFFSET + i for i in range(N_SEEDS)]
        mean_curve = []
        for a in ALPHA_GRID:
            vals = [rho_at(df, a, KFold(n_splits, shuffle=True, random_state=s)) for s in seeds]
            mean_curve.append(np.mean(vals))
        mean_curve = np.array(mean_curve)
        astar = ALPHA_GRID[mean_curve.argmax()]
        fold_rows.append({"scheme": f"{n_splits}fold_mean50seeds",
                          "alpha_star": astar, "rho_star": mean_curve.max(),
                          "rho_alpha0": mean_curve[0],
                          "delta_star_vs_0": mean_curve.max() - mean_curve[0]})
    # LOO: deterministic
    loo = LeaveOneOut()
    loo_curve = np.array([rho_at(df, a, loo) for a in ALPHA_GRID])
    astar = ALPHA_GRID[loo_curve.argmax()]
    fold_rows.append({"scheme": "LOO", "alpha_star": astar, "rho_star": loo_curve.max(),
                      "rho_alpha0": loo_curve[0], "delta_star_vs_0": loo_curve.max() - loo_curve[0]})

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(OUT_DIR / "p05e_fold_scheme_argmax.csv", index=False)

    print("===== A/B/C SIGNIFICANCE =====")
    print(summary.to_string(index=False))
    print("\n===== D FOLD-SCHEME ARGMAX =====")
    print(fold_df.to_string(index=False))


if __name__ == "__main__":
    main()

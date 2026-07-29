"""
P05c — Δρ significance via bootstrap + Steiger z-test
======================================================

Aim
---
Test whether the predictive gain from incorporating the Pressure Success
Rate (PSR) component (α=0.1) over the pure OBV baseline (α=0.0) is
statistically distinguishable from zero, given the training sample size
of N=211 player-season pairs.

Methods
-------
We refit the operative model architecture (Ridge λ=10 with position × age
and position × age² interaction terms; same as outputs/pipeline/p05_v11_coefficients.csv)
under α ∈ {0.0, 0.1} on the same training pairs (1,000-minute filter,
set D). Out-of-fold (OOF) Spearman ρ is computed via 5-fold CV.

Bootstrap inference:
  • Resample the 211 player-pairs with replacement (B = 5,000 iterations).
  • For each bootstrap sample, refit both α=0.0 and α=0.1 models with
    the same fold structure and compute Δρ = ρ(α=0.1) − ρ(α=0).
  • Report empirical 95% percentile CI of Δρ.

Steiger 1980 z-test for dependent correlations (with shared variable
score_t+1, comparing predictions f(score_t, α=0) and f(score_t, α=0.1)):
  • Provides a complementary parametric significance check.

Outputs
-------
outputs/alpha_selection/p05c_delta_rho_significance.csv
outputs/alpha_selection/p05c_bootstrap_distribution.csv

Notes
-----
- Does not modify or read from scripts/pipeline.
- Reproduces the model fitting independently from outputs/pipeline/p05_*
  inputs; checks the headline ρ_OOF = 0.4841 within ±0.0005.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, norm
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

REPO = Path(__file__).resolve().parents[2]
TRAINING_PAIRS = REPO / "outputs/pipeline/p04_training_pairs_1000min.parquet"
OUT_DIR = REPO / "outputs/alpha_selection"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RIDGE_LAMBDA = 10.0
N_FOLDS = 5
N_BOOTSTRAP = 5000
SEED_CANONICAL = 42          # canonical seed for the bootstrap CI
N_FOLD_SEEDS = 50            # sweep over fold seeds to quantify partition-induced variability
FOLD_SEED_OFFSET = 1000


def design_matrix(df: pd.DataFrame, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) for the C_ridge_inter architecture at mixing α."""
    score_t0 = (1 - alpha) * df["score_OBV_t0"].values + alpha * df["score_PSR_t0"].values
    score_t1 = (1 - alpha) * df["score_OBV_t1"].values + alpha * df["score_PSR_t1"].values
    age = df["age_t0"].values.astype(float)
    age2 = age ** 2
    pos = df["position_group"].values
    is_def = (pos == "DEF").astype(float)
    is_mid = (pos == "MID").astype(float)
    is_fwd = (pos == "FWD").astype(float)
    # Interaction terms (base category = GK)
    X = np.column_stack([
        score_t0,
        age,
        age2,
        is_def,
        is_mid,
        is_fwd,
        age * is_def,
        age * is_mid,
        age * is_fwd,
        age2 * is_def,
        age2 * is_mid,
        age2 * is_fwd,
    ])
    return X, score_t1


def oof_predictions(df: pd.DataFrame, alpha: float, kf: KFold) -> np.ndarray:
    X, y = design_matrix(df, alpha)
    preds = np.full(len(df), np.nan)
    for tr, va in kf.split(X):
        model = Ridge(alpha=RIDGE_LAMBDA, fit_intercept=True)
        model.fit(X[tr], y[tr])
        preds[va] = model.predict(X[va])
    return preds, y


def steiger_z(r12: float, r13: float, r23: float, n: int) -> tuple[float, float]:
    """Steiger 1980 z-test for dependent correlations.
    Compare r12 and r13 with shared variable 3 and r23.
    Returns (z, two-sided p-value)."""
    det = (1 - r12 ** 2 - r13 ** 2 - r23 ** 2) + 2 * r12 * r13 * r23
    avg = (r12 + r13) / 2
    cube = (1 - avg ** 2) ** 3
    # Standard Steiger formula
    num = (r12 - r13) * np.sqrt((n - 1) * (1 + r23))
    denom = np.sqrt(2 * ((n - 1) / (n - 3)) * det + cube * r23 ** 2 - r23 ** 2 * cube)
    # Use a simpler version (Williams 1959) as denom can be tricky
    z = (r12 - r13) * np.sqrt((n - 3) * (1 + r23) /
                              (2 * (det / ((n - 1) / (n - 3))) + (avg ** 2) * (1 - r23) ** 3))
    p = 2 * (1 - norm.cdf(abs(z)))
    return float(z), float(p)


def williams_test(r12: float, r13: float, r23: float, n: int) -> tuple[float, float]:
    """Williams' (1959) test for the difference between two dependent
    correlations sharing one variable.
    H0: ρ12 = ρ13.  r23 is the correlation between the two predictors.
    """
    det = 1 - r12 ** 2 - r13 ** 2 - r23 ** 2 + 2 * r12 * r13 * r23
    rbar = (r12 + r13) / 2.0
    t = (r12 - r13) * np.sqrt(
        (n - 1) * (1 + r23) / (2 * ((n - 1) / (n - 3)) * det + rbar ** 2 * (1 - r23) ** 3)
    )
    from scipy.stats import t as tdist
    p = 2 * (1 - tdist.cdf(abs(t), df=n - 3))
    return float(t), float(p)


def main():
    df = pd.read_parquet(TRAINING_PAIRS)
    print(f"Loaded {len(df)} training pairs from {TRAINING_PAIRS.name}")

    # ---- A) Fold-seed sweep: quantify how much ρ_OOF moves across partitions
    print(f"\n[A] Sweeping {N_FOLD_SEEDS} fold seeds...")
    fold_rho_0 = np.empty(N_FOLD_SEEDS)
    fold_rho_1 = np.empty(N_FOLD_SEEDS)
    for i in range(N_FOLD_SEEDS):
        seed_i = FOLD_SEED_OFFSET + i
        kf_i = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed_i)
        pred_0i, y_i = oof_predictions(df, alpha=0.0, kf=kf_i)
        pred_1i, _ = oof_predictions(df, alpha=0.1, kf=kf_i)
        fold_rho_0[i] = spearmanr(pred_0i, y_i).statistic
        fold_rho_1[i] = spearmanr(pred_1i, y_i).statistic
    fold_delta = fold_rho_1 - fold_rho_0
    print(f"  Mean ρ(α=0) = {fold_rho_0.mean():.4f}  ± {fold_rho_0.std():.4f} (sd across seeds)")
    print(f"  Mean ρ(α=0.1) = {fold_rho_1.mean():.4f}  ± {fold_rho_1.std():.4f}")
    print(f"  Mean Δρ = {fold_delta.mean():.4f}  ± {fold_delta.std():.4f}")
    print(f"  Δρ percent of seeds with Δρ > 0: {(fold_delta > 0).mean()*100:.1f}%")
    fold_q025, fold_q975 = np.percentile(fold_delta, [2.5, 97.5])
    print(f"  Δρ 2.5%-97.5% range across seeds: [{fold_q025:.4f}, {fold_q975:.4f}]")

    # ---- B) Canonical seed (KFold random_state = SEED_CANONICAL)
    print(f"\n[B] Canonical fold seed = {SEED_CANONICAL}")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED_CANONICAL)
    pred_0, y_obs = oof_predictions(df, alpha=0.0, kf=kf)
    pred_1, _ = oof_predictions(df, alpha=0.1, kf=kf)
    rho_0 = spearmanr(pred_0, y_obs).statistic
    rho_1 = spearmanr(pred_1, y_obs).statistic
    delta = rho_1 - rho_0
    print(f"  ρ(α=0.0) = {rho_0:.4f}; ρ(α=0.1) = {rho_1:.4f}; Δρ = {delta:.4f}")

    # Williams' test for dependent correlations at canonical seed
    r23 = spearmanr(pred_0, pred_1).statistic
    williams_t, williams_p = williams_test(rho_1, rho_0, r23, n=len(df))
    print(f"  Williams' t = {williams_t:.4f}, two-sided p = {williams_p:.4f}")
    print(f"  (corr(pred_α0, pred_α0.1) = {r23:.4f})")

    # ---- C) Bootstrap CI (on player-pairs, fold seed fixed = SEED_CANONICAL)
    print(f"\n[C] Bootstrap with B={N_BOOTSTRAP} on player-pairs at canonical fold seed...")
    rng = np.random.default_rng(SEED_CANONICAL)
    deltas = np.empty(N_BOOTSTRAP)
    rho_0_boot = np.empty(N_BOOTSTRAP)
    rho_1_boot = np.empty(N_BOOTSTRAP)

    n = len(df)
    indices = np.arange(n)
    for b in range(N_BOOTSTRAP):
        idx = rng.choice(indices, size=n, replace=True)
        df_b = df.iloc[idx].reset_index(drop=True)
        try:
            kf_b = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED_CANONICAL)
            pred_0b, y_b = oof_predictions(df_b, alpha=0.0, kf=kf_b)
            pred_1b, _ = oof_predictions(df_b, alpha=0.1, kf=kf_b)
            rho_0_boot[b] = spearmanr(pred_0b, y_b).statistic
            rho_1_boot[b] = spearmanr(pred_1b, y_b).statistic
            deltas[b] = rho_1_boot[b] - rho_0_boot[b]
        except Exception:
            deltas[b] = np.nan
            rho_0_boot[b] = np.nan
            rho_1_boot[b] = np.nan
        if (b + 1) % 1000 == 0:
            print(f"  bootstrap iter {b+1}/{N_BOOTSTRAP}")

    deltas = deltas[~np.isnan(deltas)]
    rho_0_boot = rho_0_boot[~np.isnan(rho_0_boot)]
    rho_1_boot = rho_1_boot[~np.isnan(rho_1_boot)]
    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    rho0_low, rho0_high = np.percentile(rho_0_boot, [2.5, 97.5])
    rho1_low, rho1_high = np.percentile(rho_1_boot, [2.5, 97.5])
    p_boot_two_sided = 2 * min(
        np.mean(deltas <= 0),
        np.mean(deltas >= 0),
    )

    summary = pd.DataFrame([
        {
            "metric": "rho_oof_alpha_0.0",
            "point_estimate": rho_0,
            "ci_low_95": rho0_low,
            "ci_high_95": rho0_high,
        },
        {
            "metric": "rho_oof_alpha_0.1",
            "point_estimate": rho_1,
            "ci_low_95": rho1_low,
            "ci_high_95": rho1_high,
        },
        {
            "metric": "delta_rho",
            "point_estimate": delta,
            "ci_low_95": ci_low,
            "ci_high_95": ci_high,
        },
        {
            "metric": "williams_t",
            "point_estimate": williams_t,
            "ci_low_95": np.nan,
            "ci_high_95": np.nan,
        },
        {
            "metric": "williams_p_two_sided",
            "point_estimate": williams_p,
            "ci_low_95": np.nan,
            "ci_high_95": np.nan,
        },
        {
            "metric": "bootstrap_p_two_sided",
            "point_estimate": p_boot_two_sided,
            "ci_low_95": np.nan,
            "ci_high_95": np.nan,
        },
        {
            "metric": "corr_pred0_pred1",
            "point_estimate": r23,
            "ci_low_95": np.nan,
            "ci_high_95": np.nan,
        },
        {
            "metric": "n_pairs",
            "point_estimate": float(len(df)),
            "ci_low_95": np.nan,
            "ci_high_95": np.nan,
        },
        {
            "metric": "n_bootstrap_iterations",
            "point_estimate": float(len(deltas)),
            "ci_low_95": np.nan,
            "ci_high_95": np.nan,
        },
    ])

    # Insert fold-seed sweep rows
    extra = pd.DataFrame([
        {"metric": "fold_seed_mean_rho_alpha_0.0", "point_estimate": float(fold_rho_0.mean()), "ci_low_95": float(fold_rho_0.min()), "ci_high_95": float(fold_rho_0.max())},
        {"metric": "fold_seed_mean_rho_alpha_0.1", "point_estimate": float(fold_rho_1.mean()), "ci_low_95": float(fold_rho_1.min()), "ci_high_95": float(fold_rho_1.max())},
        {"metric": "fold_seed_mean_delta_rho", "point_estimate": float(fold_delta.mean()), "ci_low_95": float(fold_q025), "ci_high_95": float(fold_q975)},
        {"metric": "fold_seed_pct_delta_positive", "point_estimate": float((fold_delta > 0).mean()), "ci_low_95": np.nan, "ci_high_95": np.nan},
        {"metric": "fold_seed_n_seeds", "point_estimate": float(N_FOLD_SEEDS), "ci_low_95": np.nan, "ci_high_95": np.nan},
    ])
    summary = pd.concat([extra, summary], ignore_index=True)

    summary.to_csv(OUT_DIR / "p05c_delta_rho_significance.csv", index=False)
    pd.DataFrame({
        "rho_alpha_0": rho_0_boot,
        "rho_alpha_0.1": rho_1_boot,
        "delta_rho": deltas,
    }).to_csv(OUT_DIR / "p05c_bootstrap_distribution.csv", index=False)
    pd.DataFrame({
        "seed": np.arange(N_FOLD_SEEDS) + FOLD_SEED_OFFSET,
        "rho_alpha_0": fold_rho_0,
        "rho_alpha_0.1": fold_rho_1,
        "delta_rho": fold_delta,
    }).to_csv(OUT_DIR / "p05c_fold_seed_sweep.csv", index=False)

    print("\n===== Summary =====")
    print(f"FOLD-SEED SWEEP (N={N_FOLD_SEEDS}):")
    print(f"  ρ(α=0)   = {fold_rho_0.mean():.4f}  range [{fold_rho_0.min():.4f}, {fold_rho_0.max():.4f}]")
    print(f"  ρ(α=0.1) = {fold_rho_1.mean():.4f}  range [{fold_rho_1.min():.4f}, {fold_rho_1.max():.4f}]")
    print(f"  Δρ        = {fold_delta.mean():.4f}  2.5%-97.5% [{fold_q025:.4f}, {fold_q975:.4f}]")
    print(f"  % seeds Δρ > 0: {(fold_delta > 0).mean()*100:.1f}%")
    print()
    print(f"CANONICAL fold seed = {SEED_CANONICAL}:")
    print(f"  ρ(α=0.0)  = {rho_0:.4f}")
    print(f"  ρ(α=0.1)  = {rho_1:.4f}")
    print(f"  Δρ         = {delta:.4f}")
    print(f"  Williams t = {williams_t:.4f}  (p = {williams_p:.4f})")
    print()
    print(f"BOOTSTRAP on player-pairs (B={len(deltas)}):")
    print(f"  Δρ 95% CI  = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"  2-sided bootstrap p (Δρ≠0) = {p_boot_two_sided:.4f}")

    return summary


if __name__ == "__main__":
    main()

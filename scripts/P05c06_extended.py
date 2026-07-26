"""
P05c_extended — Bootstrap inference: α* vs both pure-component baselines
========================================================================

Compares the operative blend α*=0.06 against α=0.0 (pure OBV) and
α=1.0 (pure PSR). (ALPHA_STAR is set to 0.06 below.)

All models are evaluated against the SAME target: the α*=0.06 composite
next season. This ensures both comparisons (Δρ₀ and Δρ₁) share a common
y variable, making Williams' test and bootstrap CI valid.

Point estimates from the battery (each α on its own composite target):
    ρ(α=0.0) ≈ 0.482, ρ(α*=0.06) ≈ 0.485, ρ(α=1.0) ≈ 0.361

Significance test (all evaluated on α*=0.06 composite target):
    Δρ₀ = ρ(α*) − ρ(α=0):  tests whether composite beats pure OBV
    Δρ₁ = ρ(α*) − ρ(α=1):  tests whether composite beats pure PSR

Outputs
-------
outputs/paper1_smf/p05c06_extended_summary.csv
outputs/paper1_smf/p05c06_extended_bootstrap_distribution.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as tdist
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

REPO = Path(__file__).resolve().parents[2]
TRAINING_PAIRS = REPO / "outputs/pipeline/p04_training_pairs_1000min.parquet"
OUT_DIR = REPO / "outputs/paper1_smf"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA_STAR = 0.06
RIDGE_LAMBDA = 10.0
N_FOLDS = 5
N_BOOTSTRAP = 5000
SEED_CANONICAL = 42


def build_X(df: pd.DataFrame, alpha: float) -> np.ndarray:
    """Design matrix for the operative architecture at blending weight alpha."""
    s0 = (1 - alpha) * df["score_OBV_t0"].values + alpha * df["score_PSR_t0"].values
    age = df["age_t0"].values.astype(float)
    age2 = age ** 2
    pos = df["position_group"].values
    d = (pos == "DEF").astype(float)
    m = (pos == "MID").astype(float)
    f = (pos == "FWD").astype(float)
    return np.column_stack([
        s0, age, age2, d, m, f,
        age * d, age * m, age * f,
        age2 * d, age2 * m, age2 * f,
    ])


def build_y_star(df: pd.DataFrame) -> np.ndarray:
    """Fixed target: α*=0.06 composite next season (shared ground truth)."""
    return ((1 - ALPHA_STAR) * df["score_OBV_t1"].values
            + ALPHA_STAR * df["score_PSR_t1"].values)


def oof_rho(df: pd.DataFrame, alpha: float, y_target: np.ndarray, kf: KFold) -> float:
    """OOF Spearman ρ of model(alpha) predicting y_target."""
    X = build_X(df, alpha)
    preds = np.full(len(df), np.nan)
    for tr, va in kf.split(X):
        model = Ridge(alpha=RIDGE_LAMBDA, fit_intercept=True)
        model.fit(X[tr], y_target[tr])
        preds[va] = model.predict(X[va])
    return float(spearmanr(preds, y_target).statistic)


def williams_test(r12: float, r13: float, r23: float, n: int) -> tuple[float, float]:
    det = 1 - r12 ** 2 - r13 ** 2 - r23 ** 2 + 2 * r12 * r13 * r23
    rbar = (r12 + r13) / 2.0
    t = (r12 - r13) * np.sqrt(
        (n - 1) * (1 + r23) / (2 * ((n - 1) / (n - 3)) * det + rbar ** 2 * (1 - r23) ** 3)
    )
    p = 2 * (1 - tdist.cdf(abs(t), df=n - 3))
    return float(t), float(p)


def main():
    df = pd.read_parquet(TRAINING_PAIRS)
    n = len(df)
    print(f"Loaded {n} training pairs")
    print(f"Target: α*={ALPHA_STAR} composite next season (fixed across all comparisons)\n")

    # ---- Canonical seed
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED_CANONICAL)
    y_star = build_y_star(df)

    X0   = build_X(df, 0.0)
    X01  = build_X(df, ALPHA_STAR)
    X1   = build_X(df, 1.0)

    preds_0  = np.full(n, np.nan)
    preds_01 = np.full(n, np.nan)
    preds_1  = np.full(n, np.nan)

    for tr, va in kf.split(X01):
        preds_0[va]  = Ridge(alpha=RIDGE_LAMBDA, fit_intercept=True).fit(X0[tr],  y_star[tr]).predict(X0[va])
        preds_01[va] = Ridge(alpha=RIDGE_LAMBDA, fit_intercept=True).fit(X01[tr], y_star[tr]).predict(X01[va])
        preds_1[va]  = Ridge(alpha=RIDGE_LAMBDA, fit_intercept=True).fit(X1[tr],  y_star[tr]).predict(X1[va])

    rho_0  = float(spearmanr(preds_0,  y_star).statistic)
    rho_01 = float(spearmanr(preds_01, y_star).statistic)
    rho_1  = float(spearmanr(preds_1,  y_star).statistic)
    d0 = rho_01 - rho_0
    d1 = rho_01 - rho_1

    print(f"[Canonical seed={SEED_CANONICAL}]")
    print(f"  ρ(α=0.0)   = {rho_0:.4f}")
    print(f"  ρ(α={ALPHA_STAR})  = {rho_01:.4f}")
    print(f"  ρ(α=1.0)   = {rho_1:.4f}")
    print(f"  Δρ₀ (α*−α=0) = {d0:+.4f}")
    print(f"  Δρ₁ (α*−α=1) = {d1:+.4f}")

    # Williams' test (shared y_star → valid)
    r_0_01 = float(spearmanr(preds_0,  preds_01).statistic)
    r_1_01 = float(spearmanr(preds_1,  preds_01).statistic)
    wt0, wp0 = williams_test(rho_01, rho_0, r_0_01, n)
    wt1, wp1 = williams_test(rho_01, rho_1, r_1_01, n)
    print(f"\n  Williams (α* vs α=0): t={wt0:.4f}, p={wp0:.4f}")
    print(f"  Williams (α* vs α=1): t={wt1:.4f}, p={wp1:.4f}")

    # ---- Bootstrap
    print(f"\n[Bootstrap B={N_BOOTSTRAP}]")
    rng = np.random.default_rng(SEED_CANONICAL)
    idx_arr = np.arange(n)

    r0_b  = np.empty(N_BOOTSTRAP)
    r01_b = np.empty(N_BOOTSTRAP)
    r1_b  = np.empty(N_BOOTSTRAP)
    d0_b  = np.empty(N_BOOTSTRAP)
    d1_b  = np.empty(N_BOOTSTRAP)

    for b in range(N_BOOTSTRAP):
        idx = rng.choice(idx_arr, size=n, replace=True)
        dfb = df.iloc[idx].reset_index(drop=True)
        yb  = build_y_star(dfb)
        Xb0  = build_X(dfb, 0.0)
        Xb01 = build_X(dfb, ALPHA_STAR)
        Xb1  = build_X(dfb, 1.0)
        try:
            kfb = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED_CANONICAL)
            pb0  = np.full(n, np.nan)
            pb01 = np.full(n, np.nan)
            pb1  = np.full(n, np.nan)
            for tr, va in kfb.split(Xb01):
                pb0[va]  = Ridge(alpha=RIDGE_LAMBDA, fit_intercept=True).fit(Xb0[tr],  yb[tr]).predict(Xb0[va])
                pb01[va] = Ridge(alpha=RIDGE_LAMBDA, fit_intercept=True).fit(Xb01[tr], yb[tr]).predict(Xb01[va])
                pb1[va]  = Ridge(alpha=RIDGE_LAMBDA, fit_intercept=True).fit(Xb1[tr],  yb[tr]).predict(Xb1[va])
            r0_b[b]  = spearmanr(pb0,  yb).statistic
            r01_b[b] = spearmanr(pb01, yb).statistic
            r1_b[b]  = spearmanr(pb1,  yb).statistic
            d0_b[b]  = r01_b[b] - r0_b[b]
            d1_b[b]  = r01_b[b] - r1_b[b]
        except Exception:
            r0_b[b] = r01_b[b] = r1_b[b] = d0_b[b] = d1_b[b] = np.nan
        if (b + 1) % 1000 == 0:
            print(f"  iter {b+1}/{N_BOOTSTRAP}")

    mask = ~np.isnan(d0_b)
    r0_b = r0_b[mask]; r01_b = r01_b[mask]; r1_b = r1_b[mask]
    d0_b = d0_b[mask]; d1_b  = d1_b[mask]

    ci0_lo, ci0_hi   = np.percentile(d0_b,  [2.5, 97.5])
    ci1_lo, ci1_hi   = np.percentile(d1_b,  [2.5, 97.5])
    rho0_lo,  rho0_hi  = np.percentile(r0_b,  [2.5, 97.5])
    rho01_lo, rho01_hi = np.percentile(r01_b, [2.5, 97.5])
    rho1_lo,  rho1_hi  = np.percentile(r1_b,  [2.5, 97.5])
    p0_boot = 2 * min(np.mean(d0_b <= 0), np.mean(d0_b >= 0))
    p1_boot = 2 * min(np.mean(d1_b <= 0), np.mean(d1_b >= 0))

    print(f"\n===== Final Results =====")
    print(f"ρ(α=0.0)  = {rho_0:.4f}  95% CI [{rho0_lo:.3f}, {rho0_hi:.3f}]")
    print(f"ρ(α={ALPHA_STAR})  = {rho_01:.4f}  95% CI [{rho01_lo:.3f}, {rho01_hi:.3f}]")
    print(f"ρ(α=1.0)  = {rho_1:.4f}  95% CI [{rho1_lo:.3f}, {rho1_hi:.3f}]")
    print(f"Δρ₀ = {d0:+.4f}  95% CI [{ci0_lo:.3f}, {ci0_hi:.3f}]  p(boot)={p0_boot:.4f}  Williams t={wt0:.2f} p={wp0:.4f}")
    print(f"Δρ₁ = {d1:+.4f}  95% CI [{ci1_lo:.3f}, {ci1_hi:.3f}]  p(boot)={p1_boot:.4f}  Williams t={wt1:.2f} p={wp1:.4f}")

    # Save
    summary = pd.DataFrame([
        {"metric": "n_pairs",            "value": float(n)},
        {"metric": "alpha_star",         "value": ALPHA_STAR},
        {"metric": "target",             "value": f"alpha={ALPHA_STAR} composite next season"},
        {"metric": "rho_alpha_0",        "value": rho_0,  "ci_lo": rho0_lo,  "ci_hi": rho0_hi},
        {"metric": "rho_alpha_star",     "value": rho_01, "ci_lo": rho01_lo, "ci_hi": rho01_hi},
        {"metric": "rho_alpha_1",        "value": rho_1,  "ci_lo": rho1_lo,  "ci_hi": rho1_hi},
        {"metric": "delta_rho_0",        "value": d0, "ci_lo": ci0_lo, "ci_hi": ci0_hi, "p_boot": p0_boot,
         "williams_t": wt0, "williams_p": wp0},
        {"metric": "delta_rho_1",        "value": d1, "ci_lo": ci1_lo, "ci_hi": ci1_hi, "p_boot": p1_boot,
         "williams_t": wt1, "williams_p": wp1},
    ])
    summary.to_csv(OUT_DIR / "p05c06_extended_summary.csv", index=False)
    pd.DataFrame({
        "rho_alpha_0":   r0_b,
        "rho_alpha_star": r01_b,
        "rho_alpha_1":   r1_b,
        "delta_rho_0":   d0_b,
        "delta_rho_1":   d1_b,
    }).to_csv(OUT_DIR / "p05c06_extended_bootstrap_distribution.csv", index=False)
    print(f"\nOutputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()

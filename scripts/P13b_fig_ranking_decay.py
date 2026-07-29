"""Generate the ranking-decay figure (Figure 4 of the manuscript).

Plots the Spearman rank correlation between the projected indicator at
horizon k and at k=0, over the projection panel (N = 759 players), and
marks the three decision regimes the manuscript defines.

Source: data/derived/p07_projected_scores.parquet
Outputs: figures/fig_ranking_decay.{pdf,svg}

Run from the repository root:
    python scripts/P13b_fig_ranking_decay.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "derived" / "p07_projected_scores.parquet"
OUTDIR = ROOT / "figures"

BLUE = "#1f77b4"


def ranking_decay(panel: pd.DataFrame) -> list[float]:
    """Spearman rho between the projection at horizon k and at k=0."""
    return [
        spearmanr(panel[f"score_h{k}"], panel["score_h0"]).statistic
        for k in range(6)
    ]


def main() -> None:
    panel = pd.read_parquet(PANEL)
    rho = ranking_decay(panel)
    ks = list(range(6))

    fig, ax = plt.subplots(figsize=(6.4, 3.9))

    # Decision regimes defined in the manuscript.
    ax.axvspan(-0.25, 2.5, color="#2ca02c", alpha=0.07)
    ax.axvspan(2.5, 3.5, color="#ff7f0e", alpha=0.09)
    ax.axvspan(3.5, 5.3, color="0.5", alpha=0.11)

    ax.axhline(0.7, color="0.45", lw=0.9, ls=":", zorder=1)
    ax.text(5.22, 0.715, r"$\rho = 0.7$", ha="right", va="bottom",
            fontsize=7.5, color="0.35")

    ax.plot(ks, rho, "-o", color=BLUE, lw=1.9, ms=5.5, zorder=3)

    for k, r in zip(ks, rho):
        if k == 0:
            continue
        ax.annotate(f"{r:.2f}", (k, r), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8, color=BLUE)

    ax.text(1.25, 0.06, "precise rank-order\ncomparison", ha="center",
            fontsize=7.5, color="#1a6b1a")
    ax.text(3.0, 0.06, "categorical\nplanning", ha="center",
            fontsize=7.5, color="#a35209")
    ax.text(4.5, 0.06, "not informative", ha="center",
            fontsize=7.5, color="0.35")

    ax.set_xlabel(r"Projection horizon $k$ (seasons ahead)")
    ax.set_ylabel(r"Spearman $\rho$ between rankings at $k$ and $k=0$")
    ax.set_xlim(-0.25, 5.3)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(ks)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    OUTDIR.mkdir(exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(OUTDIR / f"fig_ranking_decay.{ext}", bbox_inches="tight")

    print(f"N = {len(panel)}")
    for k, r in zip(ks, rho):
        print(f"  rho_{k} = {r:.4f}")
    print(f"written -> {OUTDIR/'fig_ranking_decay.pdf'}")


if __name__ == "__main__":
    main()

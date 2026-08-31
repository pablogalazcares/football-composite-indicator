"""P07b - Clipping audit of the recursive projection.

Summarises, per horizon, how often the projected indicator hit the bounds
of the percentile scale, together with the min, max and mean of the
projection. The manuscript reports these ranges in Supplementary S3.

This script exists because the artefact it produces previously had no
generator in the package. It was written once by hand, and when the
operative blend weight moved from alpha = 0.10 to alpha = 0.06 the panel
was regenerated and this summary was not, so the two disagreed. Deriving
it from the panel removes that failure mode: the summary cannot now be
stale relative to the file it summarises.

Input:  data/derived/p07_projected_scores.parquet
Output: data/derived/p07b_clipping_counts.csv

Run from the repository root:
    python scripts/P07b_clipping_audit.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PANEL = REPO / "data" / "derived" / "p07_projected_scores.parquet"
OUT = REPO / "data" / "derived" / "p07b_clipping_counts.csv"

# The indicator is a percentile, so the recursion is clipped to this range.
SCALE_MIN, SCALE_MAX = 0.0, 100.0
HORIZONS = range(6)


def main() -> int:
    if not PANEL.exists():
        print(f"missing input: {PANEL.relative_to(REPO)}")
        print("run P07_recursive_projection.py first")
        return 1

    panel = pd.read_parquet(PANEL)

    rows = []
    for k in HORIZONS:
        col = f"score_h{k}"
        if col not in panel.columns:
            print(f"missing column {col} in the projected panel")
            return 1
        s = panel[col]
        at_lo = int((s <= SCALE_MIN).sum())
        at_hi = int((s >= SCALE_MAX).sum())
        rows.append({
            "horizon_k": k,
            "n_players": len(s),
            "n_clipped_at_0": at_lo,
            "n_clipped_at_100": at_hi,
            "n_clipped_total": at_lo + at_hi,
            "pct_clipped": 100.0 * (at_lo + at_hi) / len(s),
            "min": s.min(),
            "max": s.max(),
            "mean": s.mean(),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(REPO)}  ({len(df)} horizons, "
          f"N = {len(panel)} players)")
    print(df[["horizon_k", "n_clipped_total", "min", "max", "mean"]]
          .round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

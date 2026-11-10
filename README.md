# Reproducibility Package — On-Ball Value Rankings One and Two Seasons Ahead

**Paper (revised version, under review)**: Galaz-Cares, P. & Sauré, D. *Ranking Premier League players one and two
seasons ahead with on-ball value: observed persistence, forecast accuracy and uncertainty*. International Journal of
Sports Science & Coaching, manuscript SPO-26-1593 (revision 1).

**DOI (data + code)**: [10.5281/zenodo.20357643](https://doi.org/10.5281/zenodo.20357643)
This is the *concept* DOI; it always resolves to the latest archived version. The version underlying the **revised**
manuscript is **v2.0.0**. Versions v1.0.0–v1.3.0 correspond to the originally submitted version (composite OBV + PSR
indicator), which the revision replaces; they are kept below for the record.

---

## v2.0.0: revision R1

The revision replaces the model-implied "ranking horizon" with **observed** rank persistence and out-of-time forecast
accuracy, using a third season (2024/25). Everything estimated from data is fixed on 2022/23 → 2023/24 and evaluated
on 2023/24 → 2024/25 (one season ahead) and 2022/23 → 2024/25 (two seasons ahead). The pressure component is dropped.

### Contents

```
paper1_ijssc/revision_R1/
  pipeline/            r01–r19 stages and run_all.py (single command, raw events → every number, table and figure)
  config/r1.yaml       seasons, paths, design constants (seed 20261108)
  tests/               60 tests (unit, data, anti-leakage, number snapshot, manuscript linters)
  PREREGISTRO*.md      time-stamped analysis plan and its two addenda (in Spanish), committed before each analysis
  outputs/d0*_results.json   read-only diagnostics of the submitted version (Supplementary S8)
  results/             numbers.tex / paper_numbers.json (every number in the paper), tables, figures, audit notes
scripts/pipeline/      P01 and P03 of the submitted pipeline (constants read, not modified, by the R1 pipeline)
data/derived/r1/       released derived data (see below)
```

### Released derived data (`data/derived/r1/`, CC BY 4.0)

Only players above the exposure threshold (≥ 1,000 minutes in covered matches) are included.

- `player_season_r1.csv` — one row per eligible player-season, 2022/23–2024/25: `season`, `player_id` (StatsBomb),
  `display_name`, `position_group` (modal position in the season), `age_on_1_january` (1 decimal), `main_club`,
  `minutes_covered`, `matches_played`, `club_matches_covered`, `share_club_minutes`, `per_action_efficiency`
  (pre-specified summary, 0–100) and `contribution_per_90` (exploratory summary, within-group percentile).
- `panel_r1_forecasts_2024_25.csv` — outfield players eligible in 2024/25 with model M2 forecasts one and two seasons
  ahead (`pred_k1`, `pred_k2`), 80% normal and conformal intervals and the probability of being in the top quartile
  of the positional group next season.

### Not included

- Raw StatsBomb event data (research-use agreement with Hudl StatsBomb; independent access required).
- Dates of birth and Transfermarkt market values (used for age and for the market-value analyses; not
  redistributed).

### Reproducing

1. Place the StatsBomb event, match and line-up JSON files where `config/r1.yaml › paths` expects them. Stages
   `r03`, `r04`, `r05b` and `r13a` also read artefacts of the submitted version (`outputs/pipeline/`, produced by the
   v1.x pipeline described below, and its player-season feature table) to check consistency with it and to document
   its errors (Supplementary S8); run the v1.x pipeline first. The analyses of the revised paper do not use its
   results.
2. Provide a players table with dates of birth (`paths.players_master`) and, for the market-value analyses,
   Transfermarkt squads and valuation histories (`paths.tm_sqlite`, `paths.tm_history`).
3. Run `python paper1_ijssc/revision_R1/pipeline/run_all.py`. The software versions used are in
   `results/environment.txt`. A clean run from the raw files reproduced all 70 outputs exactly (about 10 minutes).
4. `pytest paper1_ijssc/revision_R1/tests` checks the outputs against the released numbers.

---

## v1.x: originally submitted version (superseded)

## What this repository contains

This repository provides the complete analysis pipeline used to construct and validate the composite player-season performance indicator described in the paper. It covers the full path from raw StatsBomb event and 360 data through the projected indicator panel, together with the fold-robust selection of the OBV/PSR mixing weight.

The raw StatsBomb data are **not included** (provided under a research-use agreement with Hudl StatsBomb following the Hudl StatsBomb Conference 2024; independent StatsBomb research access required). The **derived output panel** — `data/derived/p07_projected_scores.parquet`, containing the projected composite indicator for 759 EPL players across horizons h=0…5 — is released here and is the primary reusable artefact.

---

## The mixing weight α: fold-robust selection (α = 0.06)

The composite indicator blends a normalised action-value component (OBV) and a
defensive pressure-success component (PSR):

```
score = (1 − α) · score_OBV + α · score_PSR
```

The operative weight is **α = 0.06**, selected as the fold-robust optimum of
out-of-fold (OOF) Spearman ρ. A single-partition argmax (seed = 42) sits at
α ≈ 0.10, but that maximum **does not survive fold-partition resampling** and is
not adopted. The selection rationale is *non-inferiority + construct*, not
"PSR improves prediction":

- OOF ρ at the canonical partition (α = 0.06): **0.485**; fold-robust mean over
  50 partitions: **0.457**.
- vs pure OBV (shared-target bootstrap): Δρ = **−0.006**, 95% CI
  [−0.024, +0.011], p = 0.52 → the composite is **statistically indistinguishable
  from pure OBV** in predictive accuracy.
- vs pure PSR (shared-target bootstrap): Δρ = **+0.378**, 95% CI
  [+0.169, +0.520], p < 0.001 → the composite is far superior to PSR alone.
- The robust argmax of mean OOF ρ is α = 0.06 under both 5-fold and 10-fold
  resampling (LOO: α = 0.05).

PSR is retained for its near-orthogonal defensive dimension
(ρ(OBV, PSR) ≈ 0.10) at a negligible predictive cost. The evidence tables that
back these numbers are shipped in `data/derived/alpha_selection/`.

---

## Repository structure

```
scripts/
  P01_canonical_positions.py     Canonical positional groups (GK/DEF/MID/FWD)
  P01b_events_outcomes.py        Under-pressure event coverage audit
  P02_obv_aggregation.py         OBV and PSR aggregation by player-season
  P03_score_computation.py       Component scores (score_OBV, score_PSR)
  P04_training_pairs.py          Year-on-year training pairs (set D, N=211)
  P05_model_battery.py           Model battery: architecture + biological plausibility filter
  P05c_significance.py           Δρ bootstrap + Steiger/Williams tests (PSR contribution)
  P05d_alpha_grid_sweep.py       Fine α-grid OOF sweep under 50 fold-seed partitions
  P05e_alpha_significance_and_folds.py  Significance of small-α gain + 5/10-fold/LOO robustness
  P05c06_significance_alpha006.py  Δρ inference at the operative weight α = 0.06
  P05c06_extended.py             Shared-target bootstrap: α* vs pure OBV and pure PSR
  P05z_pin_operative_alpha.py    Pin operative weight to α = 0.06; refit P05 artifacts
  P06_projection_universe.py     Projection universe C = A union B (759 players)
  P07_recursive_projection.py    Recursive 5-season projection
  P07b_clipping_audit.py         Clipping audit of the projection (Supplementary S3)
  P13a_paper1_figures.py         All figures in the paper
  P13b_fig_ranking_decay.py      Ranking-decay figure: Spearman ρ at horizon k vs k=0
  pipeline_utils.py              Shared utilities

data/
  derived/
    p07_projected_scores.parquet    Projected indicator panel (759 players, h=0 to h=5), α = 0.06
    p02b_ar1_for_against.csv        Year-on-year AR1 ρ of mean OBV by event type, position, direction
    p07b_clipping_counts.csv        [0,100] clipping events of the projection, by horizon
    p07c_profile_curves.csv         Trajectories of the 13 illustrative career-archetype players
    alpha_selection/
      p05d_alpha_grid_summary.csv       Per-α robust OOF ρ over 50 fold seeds
      p05d_argmax_per_seed.csv          Per-seed argmax of OOF ρ (clusters at α ≈ 0.06)
      p05c06_significance_summary.csv   Δρ vs OBV / vs PSR at α = 0.06 (bootstrap + Williams)
      p05c06_extended_summary.csv       Shared-target bootstrap summary
      p05e_significance_summary.csv     Per-seed paired tests (5-fold / 10-fold / bootstrap)
      p05e_fold_scheme_argmax.csv       Robust argmax under 5-fold / 10-fold / LOO
      p05c_fold_seed_sweep.csv          OOF ρ at α=0 vs α=0.06 across the 50 fold seeds
      p05c06_extended_bootstrap_distribution.csv  Full 5,000-resample shared-target bootstrap draws
  awards/
    awards_epl_2324.yaml            EPL individual awards 2023/24 (auxiliary reference data)
    awards_epl_2425.yaml            EPL individual awards 2024/25 (auxiliary reference data)
    awards_epl_2526.yaml            EPL individual awards 2025/26 (auxiliary reference data)
```

> **Note on awards files**: `data/awards/` contains manually curated records of EPL individual season awards (PFA Player of the Year, FWA Footballer of the Year, Golden Boot, etc.) used as auxiliary reference data during development. They are not part of the main analysis or any table/figure in the paper, but are released here for completeness under CC BY 4.0.

---

## Requirements

Python 3.10 or higher.

```bash
pip install -r requirements.txt
```

Dependencies: `pandas`, `numpy`, `matplotlib`, `scipy`, `scikit-learn`, `pyarrow`, `pyyaml`.

---

## How to reproduce the analysis

### If you have StatsBomb research access

Place the raw event and 360 data for EPL 2022/23 and 2023/24 in `data/raw/` following the StatsBomb open-data directory convention. Then run the pipeline in order:

```bash
python scripts/P01_canonical_positions.py
python scripts/P01b_events_outcomes.py
python scripts/P02_obv_aggregation.py
python scripts/P03_score_computation.py
python scripts/P04_training_pairs.py
python scripts/P05_model_battery.py               # architecture selection + plausibility filter
python scripts/P05c_significance.py               # PSR-contribution inference
python scripts/P05d_alpha_grid_sweep.py           # fine α-grid over 50 fold seeds
python scripts/P05e_alpha_significance_and_folds.py  # significance + 5/10-fold/LOO robustness
python scripts/P05c06_significance_alpha006.py    # inference at operative α = 0.06
python scripts/P05c06_extended.py                 # shared-target bootstrap vs OBV/PSR
python scripts/P05z_pin_operative_alpha.py        # pin α = 0.06 and refit operative artifacts
python scripts/P06_projection_universe.py
python scripts/P07_recursive_projection.py
python scripts/P07b_clipping_audit.py             # clipping audit
python scripts/P13a_paper1_figures.py
python scripts/P13b_fig_ranking_decay.py          # ranking-decay figure
```

Each script reads from `outputs/pipeline/` (prior steps) and writes its outputs there;
the α-selection scripts write their summaries to `outputs/alpha_selection/`; the
copies published with this package are under `data/derived/alpha_selection/`.
Random seed for the canonical partition: **42**; the fold-robust selection averages
over **50** fold-seed partitions (seeds 1000–1049). Operative α: **0.06**.

### If you only have the derived panel

The file `data/derived/p07_projected_scores.parquet` can be loaded directly for downstream analysis. The file is in **wide format** — one row per player, with separate columns for each projection horizon.

Schema:

| Column | Type | Description |
|---|---|---|
| `player_id` | int | StatsBomb player identifier |
| `player_name` | str | Player name |
| `position_group` | str | GK / DEF / MID / FWD |
| `age_base` | float | Age at base season (2023/24) |
| `score_pt_base` | float | Composite indicator at base season (within-position percentile, 0–100) |
| `score_h0` | float | Composite indicator at h=0 (base season; equals `score_pt_base`) |
| `score_h1` | float | Projected composite indicator at h=1 (one season ahead) |
| `score_h2` | float | Projected composite indicator at h=2 |
| `score_h3` | float | Projected composite indicator at h=3 |
| `score_h4` | float | Projected composite indicator at h=4 |
| `score_h5` | float | Projected composite indicator at h=5 |

All scores are within-position percentiles on a 0–100 scale, at α = 0.06. N=759 players (projection pool C = A ∪ B).

```python
import pandas as pd
df = pd.read_parquet("data/derived/p07_projected_scores.parquet")
# 759 rows, one per player
# score_h0 to score_h5: projected composite indicator at each horizon
print(df[["player_name", "position_group", "age_base", "score_h0", "score_h1", "score_h2"]].head())
```

---

## Key methodological parameters

| Parameter | Value | Note |
|---|---|---|
| Mixing weight α | **0.06** | Fold-robust argmax of OOF Spearman ρ (5-fold and 10-fold over 50 partitions); single-partition argmax α ≈ 0.10 is not adopted |
| Minimum minutes threshold | 1,000 min/season | Operative threshold for training set D |
| Canonical CV seed | 42 | Single-partition reference (P05c) |
| Fold-seed partitions | 50 (seeds 1000–1049) | Fold-robust α selection (P05d/P05e) |
| Bootstrap resamples (Δρ) | 5,000 | Inference on PSR predictive contribution |
| Projection horizon | 5 seasons | P07 recursive application |
| Training pairs N | 211 | EPL 2022/23 and 2023/24, set D = A intersect B |

---

## Licence

Code: MIT License (see `LICENSE`).

Derived data (`p07_projected_scores.parquet`, `p02b_*.csv`, `p07b_*.csv`, `p07c_*.csv`, `alpha_selection/*.csv`, `awards_epl_*.yaml`): released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Please cite the paper if you use this dataset.

Raw StatsBomb data are not redistributed. Documentation of the data version and processing steps is available from the corresponding author upon request.

---

## Citation

```bibtex
@unpublished{galaz-cares2026composite,
  author = {Galaz-Cares, Pablo and Sauré, Denis},
  title  = {What a projected performance indicator can and cannot tell
            a club: ranking horizons for on-ball value and pressure in
            the Premier League},
  year   = {2026},
  note   = {Under review}
}
```

---

## Contact

Pablo Galaz-Cares — pablo.galaz.c@ug.uchile.cl  
Doctoral Programme in Systems Engineering, Universidad de Chile  
ORCID: [0009-0002-1945-5702](https://orcid.org/0009-0002-1945-5702)

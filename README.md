# Reproducibility Package — Football Composite Performance Indicator

**Paper**: Galaz-Cares, P. & Sauré, D. (under review). *Measuring Technical–Tactical Performance in Football: A Composite Indicator Integrating Action Value and Defensive Pressure*. Science and Medicine in Football.

**DOI (data + code)**: [10.5281/zenodo.20357644](https://doi.org/10.5281/zenodo.20357644)

---

## What this repository contains

This repository provides the complete analysis pipeline used to construct and validate the composite player-season performance indicator described in the paper. It covers the full path from raw StatsBomb event and 360 data through the projected indicator panel.

The raw StatsBomb data are **not included** (provided under a research-use agreement with Hudl StatsBomb following the Hudl StatsBomb Conference 2024; independent StatsBomb research access required). The **derived output panel** — `data/derived/p07_projected_scores.parquet`, containing the projected composite indicator for 759 EPL players across horizons h=0…5 — is released here and is the primary reusable artefact.

---

## Repository structure

```
scripts/
  P01_canonical_positions.py     Canonical positional groups (GK/DEF/MID/FWD)
  P01b_events_outcomes.py        Under-pressure event coverage audit
  P02_obv_aggregation.py         OBV and PSR aggregation by player-season
  P03_score_computation.py       Component scores (score_OBV, score_PSR)
  P04_training_pairs.py          Year-on-year training pairs (set D, N=211)
  P05_model_battery.py           Model battery: alpha calibration + biological plausibility filter
  P06_projection_universe.py     Projection universe C = A union B (759 players)
  P07_recursive_projection.py    Recursive 5-season projection
  P13a_paper1_figures.py         All figures in the paper
  pipeline_utils.py              Shared utilities

data/
  derived/
    p07_projected_scores.parquet    Projected indicator panel (759 players, h=0 to h=5)
  awards/
    awards_epl_2324.yaml            EPL individual awards 2023/24 (source-documented)
    awards_epl_2425.yaml            EPL individual awards 2024/25
    awards_epl_2526.yaml            EPL individual awards 2025/26
```

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
python scripts/P05_model_battery.py
python scripts/P06_projection_universe.py
python scripts/P07_recursive_projection.py
python scripts/P13a_paper1_figures.py
```

Each script reads from `outputs/pipeline/` (prior steps) and writes its outputs there.
Random seed used throughout: **42**. Operative alpha: **0.10**.

### If you only have the derived panel

The file `data/derived/p07_projected_scores.parquet` can be loaded directly for downstream analysis. Schema:

| Column | Description |
|---|---|
| `player_id` | StatsBomb player identifier |
| `player_name` | Player name |
| `position_group` | GK / DEF / MID / FWD |
| `age_h0` | Age at base season (2023/24) |
| `horizon` | Projection horizon (0 = base season, 1–5 = projected) |
| `score_composite` | Composite indicator (within-position percentile, 0–100) |
| `score_OBV` | OBV component percentile |
| `score_PSR` | Pressure Success Rate component percentile |

```python
import pandas as pd
df = pd.read_parquet("data/derived/p07_projected_scores.parquet")
```

---

## Key methodological parameters

| Parameter | Value | Note |
|---|---|---|
| Mixing weight alpha | 0.10 | Selected by highest OOF Spearman rho (biologically plausible models only) |
| Minimum minutes threshold | 1,000 min/season | Operative threshold for training set D |
| Cross-validation seed | 42 | Canonical seed (P05c) |
| Bootstrap resamples (Delta rho) | 5,000 | Inference on PSR predictive contribution |
| Projection horizon | 5 seasons | P07 recursive application |
| Training pairs N | 211 | EPL 2022/23 and 2023/24, set D = A intersect B |

---

## Licence

Code: MIT License (see `LICENSE`).

Derived data (`p07_projected_scores.parquet`, `awards_epl_*.yaml`): released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Please cite the paper if you use this dataset.

Raw StatsBomb data are not redistributed. Documentation of the data version and processing steps is available from the corresponding author upon request.

---

## Citation

```bibtex
@article{galaz-cares2025composite,
  author  = {Galaz-Cares, Pablo and Sauré, Denis},
  title   = {Measuring Technical--Tactical Performance in Football:
             A Composite Indicator Integrating Action Value and Defensive Pressure},
  journal = {Science and Medicine in Football},
  year    = {2025},
  note    = {Under review}
}
```

---

## Contact

Pablo Galaz-Cares — pablo.galaz.c@ug.uchile.cl  
Doctoral Programme in Systems Engineering, Universidad de Chile  
ORCID: [0009-0002-1945-5702](https://orcid.org/0009-0002-1945-5702)

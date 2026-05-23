"""
P13a — Paper 1 Figures (performance scoring & projection)
==========================================================
Objetivo:
    Generar todas las figuras del Paper 1 desde los outputs de P01–P07.
    Este script es autónomo respecto a Paper 2: no requiere P08–P12.

    Scope: §3 (Data), §4 (Score Construction), §5 (Calibration), §6 (Projection)

Figuras generadas (en paper1_performance/figures/):
    §3/§4 — Data EDA:
        fig_obv_distribution.pdf         — OBV percentile por grupo posicional
        fig_obv_consistency.pdf          — OBV 22/23 vs 23/24 para Set D
        fig_obv_vs_minutes.pdf           — OBV vs minutos con LOWESS
        fig_f360_distributions.pdf       — 360 features por posición
        fig_f360_correlation.pdf         — Correlación 360 features
        fig_obv_psr_scatter.pdf          — score_OBV vs score_PSR (ρ, N)
        fig_age_by_set.pdf               — distribución de edad por set D/E/F
        fig_minutes_threshold.pdf        — N training pairs vs umbral (250/500/750/1000)

    §5 — Score construction & calibration:
        fig_score_distribution.pdf       — violin plots score_pt por posición/temporada
        fig_score_consistency.pdf        — score 22/23 vs 23/24 para Set D
        fig_delta_score_age.pdf          — Δscore por grupo de edad
        fig_persistence_by_position.pdf  — ρ por posición

    §6 — Score projection:
        fig_projection_convergence.pdf   — distribución proyectada por horizonte
        fig_spaghetti_all.pdf            — trayectorias individuales (todos)
        fig_spaghetti_pos.pdf            — trayectorias por posición

Tabla de valores numéricos:
    paper1_performance/tables/p13a_numeric_values.csv

Upstream:
    P01: outputs/pipeline/p01_canonical_positions.parquet
    P02: outputs/pipeline/p02_obv_player_season.parquet
         outputs/pipeline/p02_psr_player_season.parquet
    P03: outputs/pipeline/p03_score_player_season.parquet
    P04: outputs/pipeline/p04_training_pairs_1000min.parquet  (threshold operativo)
    P05: outputs/pipeline/p05_selected_config.json            (alpha*, modelo)
    P06: outputs/pipeline/p06_projection_universe.parquet
    P07: outputs/pipeline/p07_projected_scores.parquet

Correr:
    cd <repo_root>
    python scripts/pipeline/P13a_paper1_figures.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_utils import load_scores_with_pt, get_selected_alpha
from scipy.stats import spearmanr
from scipy.ndimage import uniform_filter1d

warnings.filterwarnings("ignore")

# ── Rutas ──────────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[2]

OBV_AGG_PATH  = REPO / "outputs/pipeline/p02_obv_player_season.parquet"
PSR_PATH      = REPO / "outputs/pipeline/p02_psr_player_season.parquet"
SCORES_PATH   = REPO / "outputs/pipeline/p03_score_player_season.parquet"
PAIRS_PATH    = REPO / "outputs/pipeline/p04_training_pairs_1000min.parquet"
UNIVERSE_PATH = REPO / "outputs/pipeline/p06_projection_universe.parquet"
PROJ_PATH     = REPO / "outputs/pipeline/p07_projected_scores.parquet"

FIGURES_DIR   = REPO / "paper1_performance/figures"
TABLES_DIR    = REPO / "paper1_performance/tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Umbrales operativos del paper (deben coincidir con §3 del paper)
MINUTE_THRESHOLDS = [250, 500, 750, 1000]

# ── Estilo global ─────────────────────────────────────────────────────────────
POSITION_COLORS = {"GK": "#2196F3", "DEF": "#4CAF50", "MID": "#FF9800", "FWD": "#F44336"}
POSITION_ORDER  = ["GK", "DEF", "MID", "FWD"]

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.constrained_layout.use": True,
})


def _savefig(name: str) -> None:
    path = FIGURES_DIR / f"{name}.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / f"{name}.svg", bbox_inches="tight")
    plt.close()
    print(f"  -> {path.name}")


def _lowess(x: np.ndarray, y: np.ndarray, frac: float = 0.4) -> tuple[np.ndarray, np.ndarray]:
    idx = np.argsort(x)
    xs, ys = x[idx], y[idx]
    window = max(3, int(len(xs) * frac))
    ys_smooth = uniform_filter1d(ys.astype(float), size=window, mode="nearest")
    return xs, ys_smooth


# ══════════════════════════════════════════════════════════════════════════════
# §3/§4 FIGURAS — Data EDA
# ══════════════════════════════════════════════════════════════════════════════

def fig_obv_distribution(scores: pd.DataFrame) -> None:
    data = scores[scores["score_OBV"].notna()].copy()

    fig, ax = plt.subplots(figsize=(5, 3.5))
    groups = [data[data["position_group"] == p]["score_OBV"].values
               for p in POSITION_ORDER]
    bplot = ax.boxplot(groups, labels=POSITION_ORDER, patch_artist=True,
                       medianprops=dict(color="black", linewidth=1.5),
                       whiskerprops=dict(linewidth=0.8),
                       flierprops=dict(marker=".", markersize=2, alpha=0.3))
    for patch, pos in zip(bplot["boxes"], POSITION_ORDER):
        patch.set_facecolor(POSITION_COLORS[pos])
        patch.set_alpha(0.7)
    ax.set_ylabel("OBV score (within-group percentile)")
    ax.set_xlabel("Positional group")
    ax.set_ylim(0, 100)
    _savefig("fig_obv_distribution")


def fig_obv_consistency(scores: pd.DataFrame) -> None:
    s22 = scores[scores["season_name"] == "2022/2023"][["player_id", "position_group", "score_OBV"]]
    s23 = scores[scores["season_name"] == "2023/2024"][["player_id", "score_OBV"]]
    setd = s22.merge(s23, on="player_id", suffixes=("_22", "_23"))
    setd = setd.dropna(subset=["score_OBV_22", "score_OBV_23"])

    fig, axes = plt.subplots(1, 4, figsize=(10, 3), sharey=True, sharex=True)
    for ax, pos in zip(axes, POSITION_ORDER):
        sub = setd[setd["position_group"] == pos]
        ax.boxplot(
            [sub["score_OBV_22"].values, sub["score_OBV_23"].values],
            labels=["22/23", "23/24"],
            patch_artist=True,
            boxprops=dict(facecolor=POSITION_COLORS[pos], alpha=0.6),
            medianprops=dict(color="black", linewidth=1.5),
            flierprops=dict(marker=".", markersize=2, alpha=0.3),
        )
        ax.set_title(pos)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("OBV percentile score")
    fig.suptitle("OBV score distribution: 2022/23 vs 2023/24 (Set D)", fontsize=9)
    _savefig("fig_obv_consistency")


def fig_obv_vs_minutes(scores: pd.DataFrame) -> None:
    psf_path = REPO / "data/processed/modeling/player_season_features.parquet"
    if not psf_path.exists():
        print("  [SKIP] fig_obv_vs_minutes: player_season_features.parquet no disponible")
        return
    psf = pd.read_parquet(psf_path)[["player_id", "season_name", "minutes_played"]]
    data = scores.merge(psf, on=["player_id", "season_name"], how="left")
    data = data.dropna(subset=["score_OBV", "minutes_played"])

    fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    for ax, pos in zip(axes, POSITION_ORDER):
        sub = data[data["position_group"] == pos]
        ax.scatter(sub["minutes_played"], sub["score_OBV"],
                   alpha=0.3, s=6, color=POSITION_COLORS[pos])
        xs, ys = _lowess(sub["minutes_played"].values, sub["score_OBV"].values)
        ax.plot(xs, ys, color="black", linewidth=1.5)
        for t in MINUTE_THRESHOLDS:
            ax.axvline(t, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
        r = sub[["minutes_played", "score_OBV"]].corr().iloc[0, 1]
        ax.set_title(f"{pos}  $r={r:.2f}$")
        ax.set_xlabel("Minutes played")
    axes[0].set_ylabel("OBV percentile score")
    _savefig("fig_obv_vs_minutes")


def fig_obv_psr_scatter(scores: pd.DataFrame) -> None:
    data = scores.dropna(subset=["score_OBV", "score_PSR"])
    rho, _ = spearmanr(data["score_OBV"], data["score_PSR"])

    fig, ax = plt.subplots(figsize=(4.5, 4))
    for pos in POSITION_ORDER:
        sub = data[data["position_group"] == pos]
        ax.scatter(sub["score_OBV"], sub["score_PSR"],
                   alpha=0.5, s=10, color=POSITION_COLORS[pos], label=pos)
    ax.set_xlabel("score\\_OBV (within-group percentile)")
    ax.set_ylabel("score\\_PSR (within-group percentile)")
    ax.text(0.05, 0.93, f"$\\rho = {rho:+.3f}$  ($N = {len(data)}$)",
            transform=ax.transAxes, fontsize=8)
    ax.legend(title="Position", markerscale=1.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    _savefig("fig_obv_psr_scatter")


def fig_age_by_set(universe: pd.DataFrame) -> None:
    pairs_path = PAIRS_PATH
    if pairs_path.exists():
        pairs = pd.read_parquet(pairs_path)
        set_d_ids = set(pairs["player_id"].unique())
    else:
        set_d_ids = set()

    univ = universe.copy()
    univ["canon_set"] = univ["set_label"].map({"B": "F", "E": "E"})
    univ.loc[univ["player_id"].isin(set_d_ids), "canon_set"] = "D"

    fig, ax = plt.subplots(figsize=(5, 3.5))
    for i, (lbl, color) in enumerate([("D", "#4CAF50"), ("E", "#2196F3"), ("F", "#FF9800")]):
        sub = univ[univ["canon_set"] == lbl]["age_base"].dropna()
        if len(sub) == 0:
            continue
        parts = ax.violinplot(sub, positions=[i], showmedians=True, widths=0.6)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        parts["cmedians"].set_color("black")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Set D\n(both seasons)", "Set E\n(22/23 only)", "Set F\n(23/24 only)"])
    ax.set_ylabel("Age at anchor season")
    _savefig("fig_age_by_set")


def fig_minutes_threshold() -> None:
    counts = []
    for t in MINUTE_THRESHOLDS:
        p = REPO / f"outputs/pipeline/p04_training_pairs_{t}min.parquet"
        if p.exists():
            counts.append(len(pd.read_parquet(p)))
        else:
            counts.append(np.nan)

    if all(np.isnan(c) for c in counts):
        print("  [SKIP] fig_minutes_threshold: p04 outputs no disponibles")
        return

    fig, ax = plt.subplots(figsize=(4, 3))
    bars = ax.bar(range(len(MINUTE_THRESHOLDS)),
                  counts, color="#607D8B", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(MINUTE_THRESHOLDS)))
    ax.set_xticklabels([f"{t} min" for t in MINUTE_THRESHOLDS])
    ax.set_ylabel("Eligible training pairs (Set D)")
    ax.set_xlabel("Minimum minutes threshold")
    for bar, c in zip(bars, counts):
        if not np.isnan(c):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    str(int(c)), ha="center", va="bottom", fontsize=8)
    _savefig("fig_minutes_threshold")


# ══════════════════════════════════════════════════════════════════════════════
# §5 FIGURAS — Score construction & calibration
# ══════════════════════════════════════════════════════════════════════════════

def fig_score_distribution(scores: pd.DataFrame) -> None:
    data = scores.dropna(subset=["score_pt"])

    fig, axes = plt.subplots(1, 4, figsize=(10, 3.5), sharey=True, sharex=True)
    for ax, pos in zip(axes, POSITION_ORDER):
        sub = data[data["position_group"] == pos]
        vals_22 = sub[sub["season_name"] == "2022/2023"]["score_pt"].values
        vals_23 = sub[sub["season_name"] == "2023/2024"]["score_pt"].values
        for i, (vals, color) in enumerate([(vals_22, "lightblue"), (vals_23, POSITION_COLORS[pos])]):
            if len(vals) < 2:
                continue
            parts = ax.violinplot(vals, positions=[i], showmedians=True, widths=0.7)
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            parts["cmedians"].set_color("black")
        ax.set_title(pos)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["22/23", "23/24"], fontsize=7)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("$\\mathrm{score}_{p,s}$")
    _savefig("fig_score_distribution")


def fig_score_consistency(pairs: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    for pos in POSITION_ORDER:
        sub = pairs[pairs["position_group"] == pos].dropna(subset=["score_t0", "score_t1"])
        ax.scatter(sub["score_t0"], sub["score_t1"],
                   alpha=0.5, s=14, color=POSITION_COLORS[pos], label=pos)
    ax.plot([0, 100], [0, 100], "k--", linewidth=0.8, alpha=0.5)
    valid = pairs.dropna(subset=["score_t0", "score_t1"])
    rho, _ = spearmanr(valid["score_t0"], valid["score_t1"])
    ax.set_xlabel("$\\mathrm{score}_{p,22/23}$")
    ax.set_ylabel("$\\mathrm{score}_{p,23/24}$")
    ax.text(0.05, 0.92, f"$\\rho = {rho:.2f}$  ($N = {len(valid)}$)",
            transform=ax.transAxes, fontsize=8)
    ax.legend(title="Position", markerscale=1.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    _savefig("fig_score_consistency")


def fig_delta_score_age(pairs: pd.DataFrame) -> None:
    p = pairs.dropna(subset=["score_t0", "score_t1", "age_t0"]).copy()
    p["delta"] = p["score_t1"] - p["score_t0"]
    bins   = [0, 23, 27, 30, 33, 100]
    labels = ["<24", "24–27", "28–30", "31–33", "34+"]
    p["age_group"] = pd.cut(p["age_t0"], bins=bins, labels=labels, right=False)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    means, errs, ns = [], [], []
    for lbl in labels:
        sub = p[p["age_group"] == lbl]["delta"]
        means.append(sub.mean() if len(sub) > 0 else np.nan)
        errs.append(sub.sem() * 1.96 if len(sub) > 1 else np.nan)
        ns.append(len(sub))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=errs, capsize=3, color="#607D8B", alpha=0.7,
           edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Age group at season start")
    ax.set_ylabel("Mean $\\Delta\\mathrm{score}_{p,t+1}$")
    for xi, n in zip(x, ns):
        ax.text(xi, ax.get_ylim()[1] * 0.95, f"$n={n}$", ha="center", fontsize=7)
    _savefig("fig_delta_score_age")


def fig_persistence_by_position(pairs: pd.DataFrame) -> None:
    rhos, ns = [], []
    for pos in POSITION_ORDER:
        sub = pairs[pairs["position_group"] == pos].dropna(subset=["score_t0", "score_t1"])
        if len(sub) > 4:
            r, _ = spearmanr(sub["score_t0"], sub["score_t1"])
        else:
            r = np.nan
        rhos.append(r)
        ns.append(len(sub))

    valid = pairs.dropna(subset=["score_t0", "score_t1"])
    global_rho, _ = spearmanr(valid["score_t0"], valid["score_t1"])

    fig, ax = plt.subplots(figsize=(4, 3.5))
    x = np.arange(len(POSITION_ORDER))
    ax.bar(x, rhos,
           color=[POSITION_COLORS[p] for p in POSITION_ORDER],
           alpha=0.75, edgecolor="black", linewidth=0.5)
    ax.axhline(global_rho, color="black", linestyle="--", linewidth=1,
               label=f"Global $\\rho={global_rho:.2f}$")
    ax.set_xticks(x)
    ax.set_xticklabels(POSITION_ORDER)
    ax.set_ylabel("Spearman $\\rho(\\mathrm{score}_t,\\, \\mathrm{score}_{t+1})$")
    ax.set_ylim(0, 1)
    for xi, (r, n) in enumerate(zip(rhos, ns)):
        if not np.isnan(r):
            ax.text(xi, r + 0.02, f"{r:.2f}\n$N={n}$", ha="center", fontsize=7)
    ax.legend()
    _savefig("fig_persistence_by_position")


# ══════════════════════════════════════════════════════════════════════════════
# §6 FIGURAS — Score projection
# ══════════════════════════════════════════════════════════════════════════════

def fig_projection_convergence(proj: pd.DataFrame) -> None:
    score_cols = sorted([c for c in proj.columns if c.startswith("score_h")],
                        key=lambda c: int(c.replace("score_h", "")))
    H = len(score_cols) - 1
    data = [proj[c].dropna().values for c in score_cols]

    fig = plt.figure(figsize=(11, 4))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[2, 1])

    ax1 = fig.add_subplot(gs[0])
    bplot = ax1.boxplot(data, positions=range(H + 1),
                        patch_artist=True,
                        whis=(5, 95),
                        medianprops=dict(color="black", linewidth=1.5),
                        flierprops=dict(marker="", markersize=0))
    colors = plt.cm.Blues(np.linspace(0.35, 0.85, H + 1))
    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)
    ax1.set_xlabel("Planning horizon $k$")
    ax1.set_ylabel("Projected $\\hat{\\mathrm{score}}_{p,t+k}$")
    ax1.set_ylim(0, 100)
    ax1.set_xticks(range(H + 1))

    ax2 = fig.add_subplot(gs[1])
    means = [np.nanmean(d) for d in data]
    stds  = [np.nanstd(d)  for d in data]
    ks = range(H + 1)
    ax2.plot(ks, means, "o-", color="#1565C0", label="Mean", linewidth=1.5)
    ax2.plot(ks, stds,  "s--", color="#E65100", label="Std",  linewidth=1.5)
    ax2.set_xlabel("Horizon $k$")
    ax2.legend()
    ax2.set_ylim(0, None)

    _savefig("fig_projection_convergence")


def fig_spaghetti_all(proj: pd.DataFrame, alpha: float) -> None:
    score_cols = sorted([c for c in proj.columns if c.startswith("score_h")],
                        key=lambda c: int(c.replace("score_h", "")))
    H = len(score_cols) - 1
    ks = range(H + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    for _, row in proj.iterrows():
        pos  = row.get("position_group", "MID")
        vals = [row[c] for c in score_cols]
        ax.plot(ks, vals, color=POSITION_COLORS.get(pos, "gray"),
                alpha=0.08, linewidth=0.4)

    for pos in POSITION_ORDER:
        sub = proj[proj["position_group"] == pos]
        if len(sub) == 0:
            continue
        meds = [sub[c].median() for c in score_cols]
        ax.plot(ks, meds, color=POSITION_COLORS[pos], linewidth=2,
                linestyle="--", label=pos)

    ax.set_xlabel("Planning horizon $k$")
    ax.set_ylabel("Projected $\\hat{\\mathrm{score}}_{p,t+k}$")
    ax.set_ylim(0, 100)
    ax.legend(title="Position", loc="upper right")
    ax.set_title(f"$N={len(proj)}$ players, C\\_ridge\\_inter, $\\alpha={alpha}$, $H={H}$")
    _savefig("fig_spaghetti_all")


def fig_spaghetti_pos(proj: pd.DataFrame) -> None:
    score_cols = sorted([c for c in proj.columns if c.startswith("score_h")],
                        key=lambda c: int(c.replace("score_h", "")))
    H = len(score_cols) - 1
    ks = np.arange(H + 1)

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5), sharey=True)
    for ax, pos in zip(axes, POSITION_ORDER):
        sub = proj[proj["position_group"] == pos]
        if len(sub) == 0:
            continue
        for _, row in sub.iterrows():
            vals = [row[c] for c in score_cols]
            ax.plot(ks, vals, color=POSITION_COLORS[pos], alpha=0.06, linewidth=0.3)
        q1  = sub[score_cols].quantile(0.25)
        q3  = sub[score_cols].quantile(0.75)
        med = sub[score_cols].median()
        ax.fill_between(ks, q1.values, q3.values,
                        color=POSITION_COLORS[pos], alpha=0.25)
        ax.plot(ks, med.values, color=POSITION_COLORS[pos],
                linewidth=2, linestyle="--")
        ax.set_title(f"{pos} ($N={len(sub)}$)")
        ax.set_xlabel("Horizon $k$")
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("Projected $\\hat{\\mathrm{score}}_{p,t+k}$")
    _savefig("fig_spaghetti_pos")


# ══════════════════════════════════════════════════════════════════════════════
# TABLA DE VALORES NUMÉRICOS
# ══════════════════════════════════════════════════════════════════════════════

def export_numeric_values(scores: pd.DataFrame, pairs: pd.DataFrame,
                           universe: pd.DataFrame, proj: pd.DataFrame) -> None:
    records = []

    for pos in POSITION_ORDER:
        n = (universe["position_group"] == pos).sum()
        records.append({"table": "tab_proj_universe", "row": pos, "col": "N", "value": n})
        age_m = universe[universe["position_group"] == pos]["age_base"].mean()
        records.append({"table": "tab_proj_universe", "row": pos, "col": "age_mean",
                        "value": round(age_m, 1)})

    records.append({"table": "tab_proj_universe", "row": "Total", "col": "N",
                    "value": len(universe)})

    for lbl in ["B", "E"]:
        n = (universe["set_label"] == lbl).sum()
        records.append({"table": "tab_sets_universe", "row": f"set_{lbl}", "col": "N", "value": n})

    score_cols = sorted([c for c in proj.columns if c.startswith("score_h")],
                        key=lambda c: int(c.replace("score_h", "")))
    for c in score_cols:
        h = c.replace("score_h", "")
        vals = proj[c].dropna()
        records.append({"table": "tab_proj_convergence", "row": f"h={h}", "col": "mean",
                        "value": round(vals.mean(), 1)})
        records.append({"table": "tab_proj_convergence", "row": f"h={h}", "col": "std",
                        "value": round(vals.std(), 2)})
        records.append({"table": "tab_proj_convergence", "row": f"h={h}", "col": "Q1",
                        "value": round(vals.quantile(0.25), 1)})
        records.append({"table": "tab_proj_convergence", "row": f"h={h}", "col": "Q3",
                        "value": round(vals.quantile(0.75), 1)})

    h0_col = "score_h0"
    for c in score_cols[1:]:
        h = c.replace("score_h", "")
        merged = proj[[h0_col, c]].dropna()
        if len(merged) > 4:
            rho, _ = spearmanr(merged[h0_col], merged[c])
            records.append({"table": "tab_ranking_preservation", "row": f"h={h}",
                            "col": "rho", "value": round(rho, 3)})

    for pos in POSITION_ORDER:
        sub = pairs[pairs["position_group"] == pos].dropna(subset=["score_t0", "score_t1"])
        if len(sub) > 4:
            rho, _ = spearmanr(sub["score_t0"], sub["score_t1"])
            records.append({"table": "tab_persistence", "row": pos, "col": "rho",
                            "value": round(rho, 3)})
            records.append({"table": "tab_persistence", "row": pos, "col": "N",
                            "value": len(sub)})

    df_out = pd.DataFrame(records)
    out_path = TABLES_DIR / "p13a_numeric_values.csv"
    df_out.to_csv(out_path, index=False)
    print(f"  -> {out_path.name} ({len(records)} valores)")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("P13a — Paper 1 Figures (P01–P07 scope)")
    print("=" * 60)

    required = [OBV_AGG_PATH, SCORES_PATH, UNIVERSE_PATH, PROJ_PATH]
    missing  = [p for p in required if not p.exists()]
    if missing:
        print("[ERROR] Inputs faltantes — correr P01-P07 primero:")
        for p in missing:
            print(f"  {p}")
        return

    print("Cargando datos del pipeline...")
    scores   = load_scores_with_pt(SCORES_PATH)
    universe = pd.read_parquet(UNIVERSE_PATH)
    proj     = pd.read_parquet(PROJ_PATH)
    pairs    = pd.read_parquet(PAIRS_PATH) if PAIRS_PATH.exists() else pd.DataFrame()

    # Recuperar alpha* seleccionado por P05
    try:
        alpha = get_selected_alpha()
    except Exception:
        alpha = 0.1
        print(f"  [WARN] No se pudo leer alpha* de P05 — usando {alpha}")

    # Componer score_t0/t1 en pairs si no está
    if len(pairs) > 0 and "score_t0" not in pairs.columns:
        for suf in ("t0", "t1"):
            eff_alpha = alpha * pairs[f"has_psr_{suf}"].fillna(0)
            pairs[f"score_{suf}"] = (
                (1 - eff_alpha) * pairs[f"score_OBV_{suf}"].fillna(0)
                + eff_alpha * pairs[f"score_PSR_{suf}"].fillna(0)
            ).clip(0, 100)
        print(f"  Componiendo score_t0/t1 con α={alpha}")

    # Añadir position_group al panel de proyección si no está
    if "position_group" not in proj.columns:
        proj = proj.merge(universe[["player_id", "position_group"]],
                          on="player_id", how="left")

    print(f"  Scores: {len(scores)} player-seasons")
    print(f"  Universe: {len(universe)} players")
    print(f"  Projection: {len(proj)} players × "
          f"{sum(1 for c in proj.columns if c.startswith('score_h'))} horizons")
    if len(pairs) > 0:
        print(f"  Training pairs (1000 min): {len(pairs)}")
    print(f"  alpha* = {alpha}")

    # ── §3/§4 Figuras ─────────────────────────────────────────────────────────
    print("\n§3/§4 — Data EDA figures:")
    fig_obv_distribution(scores)
    fig_obv_consistency(scores)
    fig_obv_vs_minutes(scores)
    fig_obv_psr_scatter(scores)
    fig_age_by_set(universe)
    fig_minutes_threshold()

    # ── §5 Figuras ────────────────────────────────────────────────────────────
    print("\n§5 — Score construction & calibration figures:")
    fig_score_distribution(scores)
    if len(pairs) > 0:
        fig_score_consistency(pairs)
        fig_delta_score_age(pairs)
        fig_persistence_by_position(pairs)

    # ── §6 Figuras ────────────────────────────────────────────────────────────
    print("\n§6 — Score projection figures:")
    fig_projection_convergence(proj)
    fig_spaghetti_all(proj, alpha=alpha)
    fig_spaghetti_pos(proj)

    # ── Tabla de valores numéricos ────────────────────────────────────────────
    print("\nExportando valores numéricos del paper:")
    if len(pairs) > 0:
        export_numeric_values(scores, pairs, universe, proj)

    print(f"\n[OK] Figuras en {FIGURES_DIR}")
    print(f"[OK] Valores numéricos en {TABLES_DIR}/p13a_numeric_values.csv")


if __name__ == "__main__":
    main()

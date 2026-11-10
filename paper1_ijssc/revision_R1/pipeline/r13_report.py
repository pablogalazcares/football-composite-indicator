"""R13 — Registro único de cifras, macros LaTeX, tablas y figuras (REQ-R4, T2.10).

Entradas: outputs/r1/*.json (r04–r12). Salidas en results/:
  paper_numbers.json   cifras aplanadas (clave → valor)
  numbers.tex          \\newcommand por cifra (el manuscrito no lleva literales)
  tables/*.tex         tablas booktabs
  figures/*.pdf|svg    figuras (paleta de referencia validada: azul, naranja, aqua)
  numbers_catalog.md   catálogo comentario → cifra → clave (SPEC §6.1)
"""
from __future__ import annotations

import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import Config, load_config  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
WORD = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven",
        "8": "Eight", "9": "Nine"}


def macro_name(key: str) -> str:
    parts = [p for p in re.split(r"[^A-Za-z0-9]+", key) if p]
    name = "".join(p[:1].upper() + p[1:] for p in parts)
    return "R" + "".join(WORD.get(ch, ch) for ch in name)


def fmt(v, kind: str = "r") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "NA"
    if kind == "int":
        return f"{int(round(v)):,}"
    if kind == "pct":
        return f"{100 * v:.0f}\\%"
    if kind == "r3":
        s = f"{v:.3f}"
        return s.replace("-", "$-$", 1) if s.startswith("-") else s
    if kind == "r1":
        s = f"{v:.1f}"
    else:
        s = f"{v:.2f}"
    return s.replace("-", "$-$", 1) if s.startswith("-") else s


class Registry:
    def __init__(self):
        self.vals: dict = {}
        self.fmt: dict = {}

    def add(self, key: str, value, kind: str = "r"):
        self.vals[key] = value
        self.fmt[key] = fmt(value, kind)

    def add_est(self, key: str, d: dict, kind: str = "r"):
        self.add(f"{key}.est", d["est"], kind)
        self.add(f"{key}.lo", d["ci95"][0], kind)
        self.add(f"{key}.hi", d["ci95"][1], kind)
        self.fmt[f"{key}.ci"] = f"{fmt(d['ci95'][0], kind)} to {fmt(d['ci95'][1], kind)}"  # rango del IC
        self.vals[f"{key}.ci"] = d["ci95"]

    def write(self, cfg: Config):
        cfg.res("paper_numbers.json").write_text(json.dumps(self.vals, indent=2, default=str))
        names = {}
        lines = ["% Generado por pipeline/r13_report.py. No editar a mano.", ""]
        for k, s in self.fmt.items():
            n = macro_name(k)
            if n in names:
                raise ValueError(f"macro duplicada {n}: {k} / {names[n]}")
            names[n] = k
            lines.append(f"\\newcommand{{\\{n}}}{{{s}}}  % {k}")
        cfg.res("numbers.tex").write_text("\n".join(lines) + "\n")
        return names


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_implied_vs_observed(cfg, mm, imp):
    """Solo k = 1 y 2, los horizontes observados; los valores implícitos de k > 2 invitan a extrapolar."""
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ks = np.array([1, 2])
    sub = imp["submitted_counterfactuals_from_d02"]["within_weighted_operative"][:2]
    ax.plot(ks - 0.09, sub, color=BLUE, lw=1.5, marker="o", ms=5, mec="white", mew=1,
            label="Projection with constrained age curves vs current-season score")
    iv = imp["implied_vs_observed"]
    series = (([iv["k1"]["implied_M2_W"], iv["k2"]["implied_M2_W"]], BLUE, "s", "--", "Model M2 projection vs current-season score", -0.03),
              ([iv["k1"]["observed_persistence_W"], iv["k2"]["observed_persistence_W"]], ORANGE, "o", "-",
               "Observed persistence (current-season score as forecast)", 0.03),
              ([iv["k1"]["accuracy_M2_W"], iv["k2"]["accuracy_M2_W"]], AQUA, "o", "-", "Model M2 forecast vs observed value", 0.09))
    for ser, col, mk, ls, lab, dx in series:
        y = [v["est"] for v in ser]
        lo = [v["est"] - v["ci95"][0] for v in ser]
        hi = [v["ci95"][1] - v["est"] for v in ser]
        ax.errorbar(ks + dx, y, yerr=[lo, hi], color=col, lw=1.5, ls=ls, marker=mk, ms=5,
                    mec=col if mk == "s" else "white", mfc="white" if mk == "s" else col, mew=1, capsize=0, label=lab)
    ax.set_xticks(ks)
    ax.set_xlim(0.6, 2.4)
    ax.set_xlabel("Seasons ahead (k)", color=INK, fontsize=9)
    ax.set_ylabel("Within-position Spearman ρ", color=INK, fontsize=9)
    ax.set_ylim(0, 1.05)
    _style(ax)
    ax.legend(frameon=False, fontsize=6.8, loc="lower left", labelcolor=INK)
    fig.tight_layout()
    for ext in ("pdf", "svg"):
        fig.savefig(cfg.res(f"figures/fig_implied_vs_observed.{ext}"))
    plt.close(fig)


def fig_persistence_by_position(cfg, mm):
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    rows = ["DEF", "MID", "FWD", "W"]
    labels = ["Defenders", "Midfielders", "Forwards", "All outfield (weighted)"]
    for j, (key, col, lab, dy) in enumerate((("B->C", BLUE, "One season ahead (2023/24 → 2024/25)", -0.12),
                                             ("A->C", ORANGE, "Two seasons ahead (2022/23 → 2024/25)", 0.12))):
        p = mm["persistence"][key]
        for i, r in enumerate(rows):
            d = p[f"rho_obs_{r}"]
            ax.errorbar(d["est"], i + dy, xerr=[[d["est"] - d["ci95"][0]], [d["ci95"][1] - d["est"]]], color=col,
                        marker="o", ms=5, mec="white", mew=1, lw=1.5, capsize=0, label=lab if i == 0 else None)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("Spearman ρ between seasons (95% CI)", color=INK, fontsize=9)
    ax.set_xlim(-0.2, 1)
    _style(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.4, -0.22), ncol=2, labelcolor=INK)
    fig.tight_layout()
    for ext in ("pdf", "svg"):
        fig.savefig(cfg.res(f"figures/fig_persistence_by_position.{ext}"), bbox_inches="tight")
    plt.close(fig)


def fig_reliability(cfg, unc):
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls="-")
    for key, col, lab in (("B->C", BLUE, "One season ahead"), ("A->C", ORANGE, "Two seasons ahead")):
        rel = unc["by_transition"][key]["reliability"]
        x = [r["p_mean"] for r in rel if r["n"] >= 5]
        y = [r["obs_rate"] for r in rel if r["n"] >= 5]
        ax.plot(x, y, color=col, lw=1.5, marker="o", ms=5, mec="white", mew=1, label=lab)
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 0.8)
    ax.set_xlabel("Predicted probability of top quartile", color=INK, fontsize=8)
    ax.set_ylabel("Observed share in top quartile", color=INK, fontsize=8)
    _style(ax)
    ax.legend(frameon=False, fontsize=7, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    for ext in ("pdf", "svg"):
        fig.savefig(cfg.res(f"figures/fig_reliability_top_quartile.{ext}"))
    plt.close(fig)


def fig_age_effect(cfg, fits):
    c = fits["M2_age_curve"]
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ax.fill_between(c["ages"], c["lo"], c["hi"], color=BLUE, alpha=0.12, lw=0)
    ax.plot(c["ages"], c["point"], color=BLUE, lw=1.5)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("Age on 1 January (years)", color=INK, fontsize=8)
    ax.set_ylabel("Age component vs age 26 (score points)", color=INK, fontsize=8)
    _style(ax)
    fig.tight_layout()
    for ext in ("pdf", "svg"):
        fig.savefig(cfg.res(f"figures/fig_age_effect_M2.{ext}"))
    plt.close(fig)


def fig_sensitivity(cfg, sens):
    names = [k for k in sens if k != "legacy_vs_revision_AB"]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.6), sharey=True)
    for ax, key, title in ((axes[0], "B->C", "One season ahead"), (axes[1], "A->C", "Two seasons ahead")):
        for i, n in enumerate(names):
            d = sens[n][key]["rho_obs_W"]
            ax.errorbar(d["est"], i, xerr=[[d["est"] - d["ci95"][0]], [d["ci95"][1] - d["est"]]], color=BLUE, marker="o",
                        ms=4, mec="white", mew=1, lw=1.2, capsize=0)
        ax.set_title(title, fontsize=8, color=INK)
        ax.set_xlim(-0.1, 0.9)
        _style(ax)
        ax.yaxis.grid(False)
        ax.xaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_xlabel("Within-position Spearman ρ", fontsize=8, color=INK)
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels([n.replace("_", " ") for n in names], fontsize=7, color=INK)
    axes[0].invert_yaxis()
    fig.tight_layout()
    for ext in ("pdf", "svg"):
        fig.savefig(cfg.res(f"figures/fig_sensitivity_persistence.{ext}"))
    plt.close(fig)


def _ci(d, k="r"):
    return f"{fmt(d['est'], k)} [{fmt(d['ci95'][0], k)}, {fmt(d['ci95'][1], k)}]"


def tables(cfg, mm, unc, att, ext, sens):
    T = cfg.paths["results"] / "tables"
    T.mkdir(parents=True, exist_ok=True)
    rows = [("One season, 2022/23 → 2023/24 (training)", "A->B"), ("One season, 2023/24 → 2024/25", "B->C"),
            ("Two seasons, 2022/23 → 2024/25", "A->C")]
    L = ["\\begin{tabular}{lrcccc}", "\\toprule",
         "Transition & $n$ & Defenders & Midfielders & Forwards & All outfield \\\\", "\\midrule"]
    for lab, key in rows:
        p = mm["persistence"][key]
        L.append(f"{lab} & {p['n']} & {_ci(p['rho_obs_DEF'])} & {_ci(p['rho_obs_MID'])} & {_ci(p['rho_obs_FWD'])} & "
                 f"{_ci(p['rho_obs_W'])} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (T / "tab_persistence.tex").write_text("\n".join(L) + "\n")
    L = ["\\begin{tabular}{llcccc}", "\\toprule",
         "Horizon & Predictor & $\\rho_W$ [95\\% CI] & $\\Delta\\rho_W$ vs current score & MAE & Top-quartile precision \\\\",
         "\\midrule"]
    names = {"M0": "Current score", "M1": "AR(1) + position", "M2": "+ age (pre-specified)", "M3": "+ position $\\times$ age"}
    for key, hz in (("B->C", "One season"), ("A->C", "Two seasons")):
        a = mm["accuracy"][key]
        for m in ("M0", "M1", "M2", "M3"):
            dd = "--" if m == "M0" else _ci(a[f"d_rho_{m}_W"], "r3")
            L.append(f"{hz if m == 'M0' else ''} & {names[m]} & {_ci(a[f'rho_{m}_W'])} & {dd} & {fmt(a[f'mae_{m}']['est'], 'r1')} & "
                     f"{fmt(a[f'topq_{m}']['est'])} \\\\")
        L.append("\\addlinespace")
    L += ["\\bottomrule", "\\end{tabular}"]
    (T / "tab_accuracy.tex").write_text("\n".join(L) + "\n")
    L = ["\\begin{tabular}{lcccc}", "\\toprule",
         "Quartile this season & \\multicolumn{2}{c}{One season ahead} & \\multicolumn{2}{c}{Two seasons ahead} \\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}", " & Observed & Model & Observed & Model \\\\", "\\midrule"]
    c1 = {r["q_t"]: r for r in unc["by_transition"]["B->C"]["top_quartile_given_quartile_t"]}
    c2 = {r["q_t"]: r for r in unc["by_transition"]["A->C"]["top_quartile_given_quartile_t"]}
    lab = {4: "Top", 3: "Third", 2: "Second", 1: "Bottom"}
    for q in (4, 3, 2, 1):
        L.append(f"{lab[q]} & {fmt(c1[q]['obs_rate'])} & {fmt(c1[q]['model_p'])} & {fmt(c2[q]['obs_rate'])} & {fmt(c2[q]['model_p'])} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (T / "tab_rank_uncertainty.tex").write_text("\n".join(L) + "\n")
    L = ["\\begin{tabular}{lccc}", "\\toprule", "Association & Transition & $n$ & Estimate [95\\% CI] \\\\", "\\midrule"]
    for key in ("A->B", "B->C"):
        v = ext["minutes_next"][key]
        L.append(f"Share of club minutes next season ($\\rho_W$) & {key} & {v['n']} & "
                 f"{fmt(v['rho_W'])} [{fmt(v['rho_W_ci95'][0])}, {fmt(v['rho_W_ci95'][1])}] \\\\")
    for key in ("A->B", "B->C"):
        v = ext["presence_auc"][key]
        L.append(f"Still in the league next season (AUC) & {key} & {v['n']} & "
                 f"{fmt(v['auc'])} [{fmt(v['auc_ci95'][0])}, {fmt(v['auc_ci95'][1])}] \\\\")
    for key, labm in (("log_mv_B", "Market value next season (partial $\\rho$)"), ("dlog_mv", "Change in market value (partial $\\rho$)")):
        v = ext["market_value"][key]
        L.append(f"{labm} & A->B & {v['n']} & {fmt(v['partial_spearman'])} [{fmt(v['ci95'][0])}, {fmt(v['ci95'][1])}] \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (T / "tab_external.tex").write_text("\n".join(L).replace("->", "$\\to$") + "\n")
    L = ["\\begin{tabular}{lcccc}", "\\toprule",
         " & \\multicolumn{2}{c}{One season} & \\multicolumn{2}{c}{Two seasons} \\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
         "Variant & $\\rho_W$ & $\\Delta\\rho_W$ (age model) & $\\rho_W$ & $\\Delta\\rho_W$ (age model) \\\\",
         "\\midrule"]
    for n, v in sens.items():
        if n == "legacy_vs_revision_AB":
            continue
        L.append(f"{n.replace('_', ' ')} & {_ci(v['B->C']['rho_obs_W'])} & {_ci(v['B->C']['d_rho_M2_W'], 'r3')} & "
                 f"{_ci(v['A->C']['rho_obs_W'])} & {_ci(v['A->C']['d_rho_M2_W'], 'r3')} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (T / "tab_sensitivity.tex").write_text("\n".join(L) + "\n")
    L = ["\\begin{tabular}{llrcccc}", "\\toprule",
         "Transition & Status next season & $n$ & Mean age & Aged 30+ & Mean score & Partially observed club \\\\", "\\midrule"]
    for key, rowsx in att["tables"].items():
        if "|" in key:
            continue
        for r in rowsx:
            L.append(f"{key} & {r['status'].replace('_', ' ')} & {r['n']} & {fmt(r['age_mean'], 'r1')} & {fmt(r['age_ge30'], 'pct')} & "
                     f"{fmt(r['score_mean'], 'r1')} & {fmt(r['partial_club_share'], 'pct')} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (T / "tab_attrition.tex").write_text("\n".join(L).replace("->", "$\\to$") + "\n")


def run(cfg: Config) -> dict:
    j = lambda n: json.loads(cfg.out(n).read_text())
    mm, unc, att, ext, imp, sens, fits = (j("metrics_main.json"), j("uncertainty.json"), j("attrition.json"),
                                          j("external.json"), j("implied.json"), j("sensitivity.json"), j("model_fits.json"))
    audit, mv, smp = j("data_audit.json"), j("minutes_validation.json"), j("samples_summary.json")
    R = Registry()
    # datos y muestras
    for s in cfg.labels:
        R.add(f"data.{s}.eligible", sum(smp["eligible_main_by_season_pos"][s].values()), "int")
    for key, v in smp["transitions"].items():
        R.add(f"sample.{key}.eligibleBoth", v["eligible_both"], "int")
    R.add("sample.sameSet", smp["same_set_all_seasons"], "int")
    R.add("minutes.medianAbsErrA", mv["A"]["median_abs_err_min"], "r")
    R.add("minutes.rhoTotalsA", mv["A"]["season_total_spearman"], "r3")
    R.add("minutes.rhoTotalsB", mv["B"]["season_total_spearman"], "r3")
    R.add("submitted.minutesInflationMedian", audit["minutes"]["psf_over_true_ratio_median"], "r")
    R.add("submitted.pairsTrueMinutes", audit["minutes"]["submitted_211_pairs_true_ge1000_both"], "int")
    # persistencia
    for key in ("A->B", "B->C", "A->C", "B->C|same_set", "A->C|same_set", "A->B|same_set"):
        p = mm["persistence"][key]
        base = "persist." + key.replace("->", "").replace("|same_set", ".same")
        R.add(f"{base}.n", p["n"], "int")
        for g in ("W", "P", "DEF", "MID", "FWD"):
            R.add_est(f"{base}.rho{g}", p[f"rho_obs_{g}"])
    # precisión
    for key in ("B->C", "A->C"):
        a = mm["accuracy"][key]
        base = "acc." + key.replace("->", "")
        R.add(f"{base}.n", a["n"], "int")
        for m in ("M0", "M1", "M2", "M3"):
            R.add_est(f"{base}.{m}.rhoW", a[f"rho_{m}_W"])
            R.add(f"{base}.{m}.mae", a[f"mae_{m}"]["est"], "r1")
            R.add(f"{base}.{m}.topq", a[f"topq_{m}"]["est"])
            R.add_est(f"{base}.{m}.impliedW", a[f"implied_{m}_W"])
            if m != "M0":
                R.add_est(f"{base}.{m}.dRhoW", a[f"d_rho_{m}_W"], "r3" if m in ("M2", "M3") else "r")
    # modelo
    R.add("model.nTrain", fits["n_train"], "int")
    R.add("model.betaS", fits["fits"]["M2"]["s_t"])
    R.add("model.betaS.lo", fits["M2_coef_ci95"]["s_t"][0])
    R.add("model.betaS.hi", fits["M2_coef_ci95"]["s_t"][1])
    R.add("model.ageC", fits["fits"]["M2"]["age_c"], "r3")
    R.add("model.ageC.lo", fits["M2_coef_ci95"]["age_c"][0], "r3")
    R.add("model.ageC.hi", fits["M2_coef_ci95"]["age_c"][1], "r3")
    R.add("model.ageCtwo", fits["fits"]["M2"]["age_c2"], "r3")
    R.add("model.ageCtwo.lo", fits["M2_coef_ci95"]["age_c2"][0], "r3")
    R.add("model.ageCtwo.hi", fits["M2_coef_ci95"]["age_c2"][1], "r3")
    # incertidumbre
    R.add("unc.sigma", unc["sigma_oof"], "r1")
    R.add("unc.halfWidthGauss", unc["width_gauss_k1"] / 2, "r1")
    for key in ("B->C", "A->C"):
        u = unc["by_transition"][key]
        base = "unc." + key.replace("->", "")
        R.add(f"{base}.covGauss", u["coverage_gauss"], "pct")
        R.add(f"{base}.covGauss.lo", u["coverage_gauss_ci95"][0], "pct")
        R.add(f"{base}.covGauss.hi", u["coverage_gauss_ci95"][1], "pct")
        R.add(f"{base}.covConf", u["coverage_conf"], "pct")
        R.add(f"{base}.brierSkill", u["brier_skill"])
        tq = {r["q_t"]: r for r in u["top_quartile_given_quartile_t"]}
        R.add(f"{base}.topGivenTop.obs", tq[4]["obs_rate"])
        R.add(f"{base}.topGivenTop.model", tq[4]["model_p"])
        R.add(f"{base}.topGivenBottom.obs", tq[1]["obs_rate"])
    # atrición e IPW
    for key, v in att["ipw"].items():
        base = "ipw." + key.replace("->", "")
        R.add(f"{base}.rhoW", v["rho_W_ipw"])
        R.add(f"{base}.lo", v["rho_W_ipw_ci95"][0])
        R.add(f"{base}.hi", v["rho_W_ipw_ci95"][1])
    # externo
    for key, v in ext["minutes_next"].items():
        base = "ext.minutes." + key.replace("->", "")
        R.add(f"{base}.rhoW", v["rho_W"])
        R.add(f"{base}.lo", v["rho_W_ci95"][0])
        R.add(f"{base}.hi", v["rho_W_ci95"][1])
    for key, v in ext["presence_auc"].items():
        base = "ext.auc." + key.replace("->", "")
        R.add(f"{base}.est", v["auc"])
        R.add(f"{base}.lo", v["auc_ci95"][0])
        R.add(f"{base}.hi", v["auc_ci95"][1])
    for key, v in ext["market_value"].items():
        base = "ext.mv." + key.replace("_", "")
        R.add(f"{base}.est", v["partial_spearman"])
        R.add(f"{base}.lo", v["ci95"][0])
        R.add(f"{base}.hi", v["ci95"][1])
    # implícito
    for i, v in enumerate(imp["submitted_rho_k_reproduced"]["pooled"], 1):
        R.add(f"submitted.rhoK{i}", v)
    for i, v in enumerate(imp["submitted_counterfactuals_from_d02"]["within_weighted_operative"], 1):
        R.add(f"submitted.rhoKwithin{i}", v)
    for i, v in enumerate(imp["M2_implied_rho_k_from_last_season_W"], 1):
        R.add(f"implied.M2.k{i}", v, "r3")
    # adenda 1 (contraste de volumen y valor de mercado ampliado)
    if cfg.out("addendum1.json").exists():
        ad = j("addendum1.json")
        tml = j("tm_link_report.json")
        R.add("add.tmLinkRate", tml["link_rate"], "pct")
        for s_, v in ad["corr_between_indicators"].items():
            R.add(f"add.corr.{s_}", v["rho_W"])
        for key in ("B->C", "A->C"):
            e = ad["E1"][key]
            base = "add.vol." + key.replace("->", "")
            R.add_est(f"{base}.rhoW", e["rho_obs_W"])
            for g in ("DEF", "MID", "FWD"):
                R.add_est(f"{base}.rho{g}", e[f"rho_obs_{g}"])
            R.add_est(f"{base}.dRhoMtwo", e["d_rho_M2_W"], "r3")
            R.add_est(f"add.diffPersist.{key.replace('->', '')}", ad["E3"][f"persistence_{key}"]["diff"])
        for key in ("A->B", "B->C"):
            m = ad["E3"][f"minutes_next_{key}"]
            R.add_est(f"add.minutes.vol.{key.replace('->', '')}", m["vol"])
            R.add_est(f"add.minutes.diff.{key.replace('->', '')}", m["diff"])
            a_ = ad["E1"][f"presence_auc_vol_{key}"]
            R.add(f"add.aucVol.{key.replace('->', '')}", a_["auc"])
        for key, v in ad["E2"].items():
            base = "add.mv." + key.replace("->", "")
            R.add(f"{base}.n", v["n"], "int")
            for kk in ("eff_level_next", "vol_level_next", "eff_change", "vol_change", "eff_level_now", "vol_level_now",
                       "diff_level_next", "diff_change"):
                R.add_est(f"{base}.{kk.replace('_', '')}", v[kk])
    import r13b_framing  # encuadre eficiencia frente a contribución (HC3-2)
    r13b_framing.extend(cfg, R)
    names = R.write(cfg)
    tables(cfg, mm, unc, att, ext, sens)
    fig_implied_vs_observed(cfg, mm, imp)
    fig_persistence_by_position(cfg, mm)
    fig_reliability(cfg, unc)
    fig_age_effect(cfg, fits)
    fig_sensitivity(cfg, sens)
    _catalog(cfg, R)
    return {"n_numbers": len(R.vals), "n_macros": len(names)}


def _catalog(cfg: Config, R: Registry) -> None:
    rows = [("M1", "ρk enviado reproducido; implícito de M2", ["submitted.rhoK1", "submitted.rhoK2", "implied.M2.k1", "implied.M2.k2"]),
            ("M2", "Persistencia observada a k = 1 y k = 2 (ρ_W)", ["persist.BC.rhoW.est", "persist.AC.rhoW.est"]),
            ("M2/M5", "Precisión: score actual frente al modelo con edad (Δρ_W)", ["acc.BC.M2.dRhoW.est", "acc.AC.M2.dRhoW.est"]),
            ("M3", "Efectos de edad de M2 con IC", ["model.ageC", "model.ageC.lo", "model.ageC.hi"]),
            ("M7", "Minutos inflados en lo enviado; IPW", ["submitted.minutesInflationMedian", "submitted.pairsTrueMinutes", "ipw.BC.rhoW", "ipw.AC.rhoW"]),
            ("M9", "Validez externa", ["ext.minutes.BC.rhoW", "ext.auc.BC.est", "ext.mv.logmvB.est"]),
            ("m2", "ρ por posición", ["persist.BC.rhoDEF.est", "persist.BC.rhoMID.est", "persist.BC.rhoFWD.est"]),
            ("m3", "Cobertura de intervalos y P(cuartil superior)", ["unc.BC.covGauss", "unc.BC.topGivenTop.obs", "unc.BC.topGivenTop.model"])]
    L = ["# Catálogo de cifras (SPEC §6.1)", "", "Las cifras sobre la versión enviada están en `evidence_submitted_version.md`.",
         "", "| Comentario | Qué | Clave → macro → valor |", "|---|---|---|"]
    for com, what, keys in rows:
        L.append(f"| {com} | {what} | " + "<br>".join(f"`{k}` → `\\{macro_name(k)}` → {R.fmt[k]}" for k in keys) + " |")
    cfg.res("numbers_catalog.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    print(run(load_config()))

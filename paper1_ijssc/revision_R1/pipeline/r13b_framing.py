"""R13b — Salidas para el encuadre "eficiencia por acción frente a contribución total" (HC3-2, 2026-11-09).

r13_report.run() llama a `extend(cfg, R)` antes de escribir el registro. Agrega:
  - macros extra: diseño (desde la config y el manifiesto), datos, sensibilidad, arqueros, estratos, atrición,
    under_pressure, enlace con Transfermarkt y cifras de la versión ENVIADA (d01–d07) para la respuesta y el S8;
  - results/design_numbers.tex (constantes de diseño);
  - tablas: tab_persistence_both, tab_accuracy_both, tab_external_both, tab_coverage, tab_ar1_weights, tab_models,
    tab_strata, tab_implied_counterfactuals, tab_under_pressure, tab_gk;
  - figura: fig_persistence_two_summaries.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import R1, Config, p03  # noqa: E402

BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
ARROW = " $\\to$ "
POSN = {"DEF": "Defenders", "MID": "Midfielders", "FWD": "Forwards", "W": "All outfield"}


def num(v, k="r"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "NA"
    s = {"r": f"{v:.2f}", "r3": f"{v:.3f}", "r1": f"{v:.1f}", "int": f"{int(round(v)):,}"}[k]
    return s.replace("-", "$-$", 1) if s.startswith("-") else s


def est_ci(d, k="r"):
    return f"{num(d['est'], k)} [{num(d['ci95'][0], k)}, {num(d['ci95'][1], k)}]"


def j(cfg, name, base="out"):
    p = cfg.out(name) if base == "out" else R1 / "outputs" / name
    return json.loads(p.read_text()) if p.exists() else None


def design_numbers(cfg: Config) -> None:
    man = j(cfg, "ingest_manifest.json")
    cov = pd.read_csv(cfg.out("coverage_by_club_season.csv"))
    a = cov[cov.season == "A"].matches_covered
    ex = cfg.raw["exposure"]
    L = ["% Generado por pipeline/r13b_framing.py desde config/r1.yaml y el manifiesto de ingesta. No editar.",
         f"\\newcommand{{\\MinMinutes}}{{{ex['main_min_minutes']:,}}}",
         f"\\newcommand{{\\SensMinutesLow}}{{{ex['sensitivity_min_minutes'][0]:,}}}",
         f"\\newcommand{{\\SensMinutesMid}}{{{ex['sensitivity_min_minutes'][1]:,}}}",
         f"\\newcommand{{\\SensShareClub}}{{{int(100 * ex['sensitivity_share_club_minutes'])}\\%}}",
         "\\newcommand{\\NMatchesSeason}{380}",
         f"\\newcommand{{\\NMatchesAB}}{{{man['A']['matches_with_events']}}}",
         f"\\newcommand{{\\NMatchesC}}{{{man['C']['matches_with_events']}}}",
         f"\\newcommand{{\\NClubsFullAB}}{{{int((a == 38).sum())}}}",
         f"\\newcommand{{\\NMatchesPartialAB}}{{{int(a[a < 38].min())}}}",
         f"\\newcommand{{\\NBoot}}{{{cfg.raw['metrics']['bootstrap_B']:,}}}",
         f"\\newcommand{{\\NSims}}{{{cfg.raw['metrics']['n_sims_rank']:,}}}",
         f"\\newcommand{{\\IntervalLevel}}{{{int(100 * cfg.raw['metrics']['interval_level'])}\\%}}",
         f"\\newcommand{{\\AgeCentre}}{{{int(cfg.raw['models']['age_center'])}}}",
         "\\newcommand{\\NFolds}{five}",
         f"\\newcommand{{\\SeedValue}}{{{cfg.seed}}}"]
    cfg.res("design_numbers.tex").write_text("\n".join(L) + "\n")


def _table(cfg, name, lines):
    p = cfg.paths["results"] / "tables" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n")


def tables_and_figure(cfg: Config) -> None:
    mm, ad, ext = j(cfg, "metrics_main.json"), j(cfg, "addendum1.json"), j(cfg, "external.json")
    # persistencia, ambos resúmenes
    # el resumen va como fila de grupo (no como columna) para que la tabla quepa en el ancho de texto
    L = ["\\begin{tabular}{lcccc}", "\\toprule",
         "Horizon ($n$) & Defenders & Midfielders & Forwards & All outfield ($\\rho_{\\mathrm{W}}$) \\\\", "\\midrule"]
    for lab, src in (("Per-action efficiency", "eff"), ("Contribution per 90 minutes", "vol")):
        L.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{lab}}}}} \\\\")
        for key, hz in (("B->C", "One season"), ("A->C", "Two seasons")):
            if src == "eff":
                p = mm["persistence"][key]
                cells = [est_ci(p[f"rho_obs_{g}"]) for g in ("DEF", "MID", "FWD", "W")]
                n = p["n"]
            else:
                p = ad["E1"][key]
                cells = [est_ci(p[f"rho_obs_{g}"]) for g in ("DEF", "MID", "FWD", "W")]
                n = p["n"]
            L.append(f"{hz} ({n}) & " + " & ".join(cells) + " \\\\")
        L.append("\\addlinespace")
    d2 = ad["E3"]["persistence_A->C"]["diff"]
    L += ["\\multicolumn{4}{l}{Difference, contribution $-$ efficiency, two seasons (paired)} & " + est_ci(d2) + " \\\\",
          "\\bottomrule", "\\end{tabular}"]
    _table(cfg, "tab_persistence_both.tex", L)
    # precisión, ambos resúmenes: M0–M3, Δρ_W frente a M0, MAE y precisión del cuartil superior (todo preespecificado)
    rob = j(cfg, "robustness.json")
    L = ["\\begin{tabular}{lcccc}", "\\toprule",
         "Forecast & $\\rho_{\\mathrm{W}}$ & $\\Delta\\rho_{\\mathrm{W}}$ vs current value & MAE & Top-quartile precision \\\\",
         "\\midrule"]
    names = {"M0": "Current value (M0)", "M1": "AR(1) + position (M1)", "M2": "M1 + age (M2)", "M3": "M2 by position (M3)"}
    for slab, src in (("Per-action efficiency", "eff"), ("Contribution per 90 minutes", "vol")):
        for key, hz in (("B->C", "one season ahead"), ("A->C", "two seasons ahead")):
            a = mm["accuracy"][key] if src == "eff" else rob["R0_completed"]["vol_accuracy"][key]
            L.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{slab}, {hz}}} ($n$ = {a['n']})}} \\\\")
            for m in ("M0", "M1", "M2", "M3"):
                dd = "--" if m == "M0" else ("same ranks as M0" if m == "M1" else est_ci(a[f"d_rho_{m}_W"], "r3"))
                L.append(f"{names[m]} & {est_ci(a[f'rho_{m}_W'])} & {dd} & {num(a[f'mae_{m}']['est'], 'r1')} & "
                         f"{est_ci(a[f'topq_{m}'])} \\\\")
            L.append("\\addlinespace")
    L = L[:-1] + ["\\bottomrule", "\\end{tabular}"]
    _table(cfg, "tab_accuracy_both.tex", L)
    # asociaciones externas
    L = ["\\begin{tabular}{lcc}", "\\toprule", "Outcome and transition & Per-action efficiency & Contribution per 90 \\\\",
         "\\midrule"]
    tlab = {"A->B": "2022/23 $\\to$ 2023/24", "B->C": "2023/24 $\\to$ 2024/25", "C->D": "2024/25 $\\to$ 2025/26"}
    L.append("\\multicolumn{3}{l}{\\textit{Share of club minutes next season}} \\\\")
    for key in ("A->B", "B->C"):
        m = ad["E3"][f"minutes_next_{key}"]
        L.append(f"{tlab[key]} & {est_ci(m['eff'])} & {est_ci(m['vol'])} \\\\")
    L.append("\\addlinespace")
    L.append("\\multicolumn{3}{l}{\\textit{Still in the league next season (AUC)}} \\\\")
    for key in ("A->B", "B->C"):
        e, v = ext["presence_auc"][key], ad["E1"][f"presence_auc_vol_{key}"]
        L.append(f"{tlab[key]} & "
                 f"{num(e['auc'])} [{num(e['auc_ci95'][0])}, {num(e['auc_ci95'][1])}] & {num(v['auc'])} [{num(v['ci95'][0])}, {num(v['ci95'][1])}] \\\\")
    L.append("\\addlinespace")
    for kk, lab in (("level_next", "Market value, end of next season"), ("level_now", "Market value, end of same season"),
                    ("change", "Change in market value")):
        L.append(f"\\multicolumn{{3}}{{l}}{{\\textit{{{lab}}}}} \\\\")
        for key in ("A->B", "B->C", "C->D"):
            v = ad["E2"][key]
            L.append(f"{tlab[key]} & {est_ci(v['eff_' + kk])} & {est_ci(v['vol_' + kk])} \\\\")
        L.append("\\addlinespace")
    L += ["\\bottomrule", "\\end{tabular}"]
    _table(cfg, "tab_external_both.tex", L)
    # figura: persistencia por posición, dos resúmenes, dos horizontes
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), sharey=True)
    rows = ["DEF", "MID", "FWD", "W"]
    for ax, key, title in ((axes[0], "B->C", "One season ahead"), (axes[1], "A->C", "Two seasons ahead")):
        for src, col, lab, dy in (("eff", BLUE, "Per-action efficiency", -0.13), ("vol", ORANGE, "Contribution per 90 minutes", 0.13)):
            p = mm["persistence"][key] if src == "eff" else ad["E1"][key]
            for i, r in enumerate(rows):
                d = p[f"rho_obs_{r}"]
                ax.errorbar(d["est"], i + dy, xerr=[[d["est"] - d["ci95"][0]], [d["ci95"][1] - d["est"]]], color=col,
                            marker="o", ms=4.5, mec="white", mew=1, lw=1.4, capsize=0, label=lab if i == 0 else None)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title(title, fontsize=8.5, color=INK)
        ax.set_xlim(-0.2, 0.95)
        ax.spines[["top", "right"]].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=7.5)
        ax.xaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("Spearman ρ (95% CI)", fontsize=8, color=INK)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([POSN[r] for r in rows], fontsize=8, color=INK)
    axes[0].invert_yaxis()
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(0.55, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ext_ in ("pdf", "svg"):
        fig.savefig(cfg.res(f"figures/fig_persistence_two_summaries.{ext_}"))
    plt.close(fig)


def supplement_tables(cfg: Config) -> None:
    cov = pd.read_csv(cfg.out("coverage_by_club_season.csv"))
    man = j(cfg, "ingest_manifest.json")
    L = ["\\begin{tabular}{lccc}", "\\toprule", "Season & Matches with event data & Clubs with all matches & Clubs with partial coverage \\\\", "\\midrule"]
    for s, nm in (("A", "2022/23"), ("B", "2023/24"), ("C", "2024/25")):
        c = cov[cov.season == s]
        full = int((c.matches_covered == 38).sum())
        part = c[c.matches_covered < 38].matches_covered.value_counts().sort_index()
        L.append(f"{nm} & {man[s]['matches_with_events']} & {full} & " + ", ".join(f"{v} with {k}" for k, v in part.items()) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    _table(cfg, "tab_coverage.tex", L)
    P3 = p03()
    L = ["\\begin{tabular}{llcc}", "\\toprule", "Group & Event type & Created (for) & Conceded (against) \\\\", "\\midrule"]
    for g in ("GK", "DEF", "MID", "FWD"):
        f, a = P3.AR1_WEIGHTS_FOR[g], P3.AR1_WEIGHTS_AGAINST[g]
        for i, e in enumerate(sorted(set(f) | set(a))):
            L.append(f"{g if i == 0 else ''} & {e} & {num(f[e]) if e in f else '--'} & {num(a[e]) if e in a else '--'} \\\\")
        L.append("\\addlinespace")
    L += ["\\bottomrule", "\\end{tabular}"]
    _table(cfg, "tab_ar1_weights.tex", L)
    mf = j(cfg, "model_fits.json")
    ci = mf["M2_coef_ci95"]
    L = ["\\begin{tabular}{lcc}", "\\toprule", "Term & M2 (95\\% bootstrap CI) & M3 \\\\", "\\midrule"]
    lab = {"intercept": "Intercept", "s_t": "Current value", "pos_MID": "Midfielder (vs defender)", "pos_FWD": "Forward (vs defender)",
           "age_c": "Age $-$ 26", "age_c2": "(Age $-$ 26)$^2$", "age_c_x_MID": "(Age $-$ 26) $\\times$ midfielder",
           "age_c2_x_MID": "(Age $-$ 26)$^2$ $\\times$ midfielder", "age_c_x_FWD": "(Age $-$ 26) $\\times$ forward",
           "age_c2_x_FWD": "(Age $-$ 26)$^2$ $\\times$ forward"}
    m2, m3 = mf["fits"]["M2"], mf["fits"]["M3"]
    for k_ in lab:
        c2 = (f"{num(m2[k_], 'r3')} [{num(ci[k_][0], 'r3')}, {num(ci[k_][1], 'r3')}]" if k_ in ci else num(m2[k_], 'r3')) if k_ in m2 else "--"
        L.append(f"{lab[k_]} & {c2} & {num(m3[k_], 'r3') if k_ in m3 else '--'} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    _table(cfg, "tab_models.tex", L)
    att = j(cfg, "attrition.json")
    L = ["\\begin{tabular}{llcc}", "\\toprule", "Transition & Club coverage in first season & $n$ & $\\rho_{\\mathrm{W}}$ (95\\% CI) \\\\", "\\midrule"]
    for k_, v in att["coverage_strata"].items():
        t, s = k_.split("|")
        tt = t.replace("->", ARROW)
        cov_lab = "partial (20 matches)" if s == "partial_20" else "full (38 matches)"
        L.append(f"{tt} & {cov_lab} & {v['n']} & "
                 f"{num(v['rho_W'])} [{num(v['ci95'][0])}, {num(v['ci95'][1])}] \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    _table(cfg, "tab_strata.tex", L)
    d02 = j(cfg, "d02_results.json", base="diag")
    cf = d02["C_counterfactual_rho"]
    L = ["\\begin{tabular}{lccccc}", "\\toprule", "Projection (submitted model) & $k=1$ & $k=2$ & $k=3$ & $k=4$ & $k=5$ \\\\", "\\midrule"]
    rows = [("As submitted, pooled", d02["B_rho_operative"]["pooled"]), ("As submitted, within group", d02["B_rho_operative"]["within_weighted"]),
            ("No age terms, within group", cf["no_age_terms"]["within_weighted"]), ("Age terms $\\times$ 0.25, pooled", cf["age_scale_0.25"]["pooled"]),
            ("Age terms $\\times$ 2, pooled", cf["age_scale_2.0"]["pooled"]), ("$\\beta_s$ = 0.3, pooled", cf["beta_0.3"]["pooled"]),
            ("$\\beta_s$ = 0.8, pooled", cf["beta_0.8"]["pooled"])]
    for lab_, vals in rows:
        L.append(f"{lab_} & " + " & ".join(num(x) for x in vals) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    _table(cfg, "tab_implied_counterfactuals.tex", L)
    au = j(cfg, "data_audit.json")
    up = au["under_pressure_by_type"]
    L = ["\\begin{tabular}{lccc}", "\\toprule", "Event type & 2022/23 & 2023/24 & 2024/25 \\\\", "\\midrule"]
    for t in ("Pass", "Carry", "Shot", "Dribble", "Duel", "Clearance"):
        L.append(f"{t} & " + " & ".join(num(up[s][t]) for s in ("A", "B", "C")) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    _table(cfg, "tab_under_pressure.tex", L)
    gk = j(cfg, "goalkeepers.json")
    if gk:
        L = ["\\begin{tabular}{llcc}", "\\toprule", "Summary & Transition & $n$ & Spearman $\\rho$ (95\\% CI) \\\\", "\\midrule"]
        for k_, v in gk.items():
            sc, t = k_.split("|")
            tt = t.replace("->", ARROW)
            slab = "Efficiency" if sc == "score_obv_ar1" else "Contribution"
            L.append(f"{slab} & {tt} & {v['n']} & {est_ci(v['rho'])} \\\\")
        L += ["\\bottomrule", "\\end{tabular}"]
        _table(cfg, "tab_gk.tex", L)


def extra_macros(cfg: Config, R) -> None:
    man, mv = j(cfg, "ingest_manifest.json"), j(cfg, "minutes_validation.json")
    R.add("minutes.nPlayerMatchesA", mv["A"]["player_matches_events"], "int")
    R.add("minutes.nPlayerMatchesB", mv["B"]["player_matches_events"], "int")
    st = j(cfg, "players_season_stats.json")
    pc = st["position_changes"]
    R.add("pos.changed.AB", pc["A->B"]["eligible_changed"], "int")
    R.add("pos.eligibleBoth.AB", pc["A->B"]["eligible_both"], "int")
    R.add("pos.changed.BC", pc["B->C"]["eligible_changed"], "int")
    R.add("pos.eligibleBoth.BC", pc["B->C"]["eligible_both"], "int")
    tml = j(cfg, "tm_link_report.json")
    R.add("tm.linked", tml["status"].get("linked", 0), "int")
    R.add("tm.total", tml["sb_players"], "int")
    sens = j(cfg, "sensitivity.json")
    v = [x for k_, x in sens.items() if k_ != "legacy_vs_revision_AB"]
    for key, lab in (("B->C", "k1"), ("A->C", "k2")):
        vals = [x[key]["rho_obs_W"]["est"] for x in v]
        R.add(f"sens.{lab}.min", min(vals))
        R.add(f"sens.{lab}.max", max(vals))
        dm = [x[key]["d_rho_M2_W"]["est"] for x in v]
        R.add(f"sens.{lab}.dmax", max(dm), "r3")
        R.add(f"sens.{lab}.dmin", min(dm), "r3")
    dci = [x[key]["d_rho_M2_W"]["ci95"] for x in v for key in ("B->C", "A->C")]
    R.add("sens.nDSigNeg", sum(c[1] < 0 for c in dci), "int")
    R.add("sens.nDSigPos", sum(c[0] > 0 for c in dci), "int")
    R.add("sens.nVariants", len(v), "int")
    R.add("sens.equalWeights.k1", sens["equal_weights"]["B->C"]["rho_obs_W"]["est"])
    R.add("sens.legacyAB", sens["legacy_vs_revision_AB"]["legacy_rho_W"]["est"])
    att = j(cfg, "attrition.json")
    ra = att["tables"]["A->B|retention_by_age"]
    R.add("att.retention.AB.young", ra["<=23"], "pct")
    R.add("att.retention.AB.old", ra["33+"], "pct")
    t = {r["status"]: r for r in att["tables"]["A->B"]}
    R.add("att.AB.absent.age30", t["absent"]["age_ge30"], "pct")
    R.add("att.AB.eligible.age30", t["eligible"]["age_ge30"], "pct")
    gk = j(cfg, "goalkeepers.json")
    if gk:
        for k_, v_ in gk.items():
            sc, tt = k_.split("|")
            R.add(f"gk.{'eff' if sc == 'score_obv_ar1' else 'vol'}.{tt.replace('->', '')}.n", v_["n"], "int")
            R.add_est(f"gk.{'eff' if sc == 'score_obv_ar1' else 'vol'}.{tt.replace('->', '')}", v_["rho"])
    au = j(cfg, "data_audit.json")["under_pressure_by_type"]
    for typ in ("Pass", "Carry", "Shot"):
        for s in ("A", "B", "C"):
            R.add(f"up.{typ}.{s}", au[s][typ], "pct")
    # cifras de la versión ENVIADA (diagnósticos d01–d07), para la respuesta y el S8
    d = {k_: j(cfg, f"{k_}_results.json", base="diag") for k_ in ("d01", "d02", "d03", "d04", "d05", "d06", "d07")}
    ref = d["d02"]["C_refits"]["operative_published"]
    for p in ("GK", "DEF", "MID", "FWD"):
        R.add(f"sub.agePeak.{p}", ref[f"age_function_peak_{p}"], "r1")
        R.add(f"sub.recPeak.{p}", ref[f"recursive_peak_{p}_start50"], "int")
        R.add(f"sub.recPeakThirty.{p}", ref[f"recursive_peak_{p}_start30"], "int")
    R.add("sub.betaRidge", d["d02"]["C_refits"]["ridge_raw"]["beta_s"], "r3")
    R.add("sub.betaUnpen", d["d02"]["C_refits"]["ridge_beta_unpen"]["beta_s"], "r3")
    R.add("sub.betaOLS", d["d02"]["C_refits"]["ols"]["beta_s"], "r3")
    for i, x in enumerate(d["d02"]["C_refits"]["ols"]["rho"]["pooled"][:3], 1):
        R.add(f"sub.olsRho.k{i}", x)
    R.add("sub.naive", d["d03"]["1_baselines_211"]["alpha_0.06"]["naive_current_score"], "r3")
    R.add("sub.oofOperative", d["d03"]["1_baselines_211"]["alpha_0.06"]["oof_operative"]["mean"], "r3")
    R.add("sub.nestedOuter", d["d03"]["2_nested_cv"]["outer_rho_vs_alpha006_target_mean"])
    R.add("sub.nestedInner", d["d03"]["2_nested_cv"]["apparent_best_inner_rho_mean"])
    R.add("sub.mddAssumed", d["d03"]["3_mdd"]["paper_formula_r23_assumed_0.95"], "r3")
    R.add("sub.mddEmpirical", d["d03"]["3_mdd"]["paper_formula_r23_empirical"], "r3")
    R.add("sub.mddBoot", d["d03"]["3_mdd"]["mdd_from_bootstrap_se_(za+zb)*se"], "r3")
    R.add("sub.rTwoThreeEmp", d["d03"]["3_mdd"]["r23_empirical_oof_predictions"], "r3")
    R.add("sub.rhoThreeMin", d["d04"]["rho3_range_all_fits"][0])
    R.add("sub.rhoThreeMax", d["d04"]["rho3_range_all_fits"][1])
    R.add("sub.gkPeakMin", min(d["d04"]["rec_peak_GK_values"]), "int")
    R.add("sub.gkPeakMax", max(d["d04"]["rec_peak_GK_values"]), "int")
    R.add("sub.nPassFilter", d["d04"]["n_pass_filter"], "int")
    R.add("sub.nFits", d["d04"]["n_fits"], "int")
    fw = d["d05"]["foldwise_alpha0.06"]
    R.add("sub.oofFixed", fw["oof_rho_fixed_weights_mean"])
    R.add("sub.oofFoldwise", fw["oof_rho_foldwise_weights_mean"])
    R.add("sub.yFixed", fw["test_fold_y2y_fixed_mean"])
    R.add("sub.yFoldwise", fw["test_fold_y2y_foldwise_mean"])
    R.add("sub.yEqual", d["d05"]["rho_y2y_alpha0.0"]["equal_event_weights"])
    p6 = d["d06"]["2_persistence"]
    R.add_est("sub.psrCode", {"est": p6["code_scorePSR|training_211"]["rho"], "ci95": p6["code_scorePSR|training_211"]["ci95"]})
    R.add_est("sub.psrEqTwo", {"est": p6["eq2_pooled_percentile|training_211"]["rho"], "ci95": p6["eq2_pooled_percentile|training_211"]["ci95"]})
    em = d["d06"]["3_event_mix"]
    R.add("sub.psrPassDEF", em["psr_pooled_vs_share_Pass_DEF"])
    R.add("sub.psrPassMID", em["psr_pooled_vs_share_Pass_MID"])
    R.add("sub.psrCodeVsEq", d["d06"]["1_code_vs_eq2"]["spearman_scorePSR_code_vs_pooled_percentile"])
    lk = d["d01"]["linkage"]
    R.add("sub.linkExact", lk["exact"], "int")
    R.add("sub.linkNames", lk["event_names"], "int")
    prof = {p_["player_name"]: p_ for p_ in d["d01"]["profile_players"]}
    R.add("sub.ottoEvents", prof["Jonathan Castro Otto"]["n_events"], "int")
    R.add("sub.ottoMinutes", prof["Jonathan Castro Otto"]["minutes_played"], "int")
    R.add("sub.romeuMinutes", prof["Oriol Romeu Vidal"]["minutes_played"], "int")
    pu = d["d01"]["projection_universe_minutes"]
    R.add("sub.universeLtThousand", pu["lt_1000"], "int")
    R.add("sub.universeBaseA", d["d01"]["projection_universe_base_season"]["2022/2023"], "int")
    R.add("sub.posChanged", d["d01"]["position_group_change_between_seasons"]["changed_group"], "int")
    R.add("sub.posBoth", d["d01"]["position_group_change_between_seasons"]["players_in_both_seasons"], "int")
    # caso citado por R2-M9: Phil Foden (StatsBomb 4354), premios PFA y FWA 2023/24; puesto entre delanteros elegibles
    tr = pd.read_parquet(cfg.out("transitions.parquet"))
    fw = tr[(tr.t == "B") & (tr.tk == "C") & tr.eligible_main_t & (tr.position_group_season_t == "FWD")]
    R.add("foden.nFwdB", len(fw), "int")
    for sc, lab in (("score_obv_ar1_t", "Eff"), ("score_obv_p90_t", "Vol")):
        rk = fw[sc].rank(ascending=False, method="min")
        R.add(f"foden.rank{lab}B", rk[fw.player_id == 4354].iloc[0], "int")


def robustness(cfg: Config, R) -> None:
    """Macros y tablas de la adenda 2 (r19) y de cantidades preespecificadas que no estaban en el texto."""
    rob, mm, unc = j(cfg, "robustness.json"), j(cfg, "metrics_main.json"), j(cfg, "uncertainty.json")
    ext, att, d07 = j(cfg, "external.json"), j(cfg, "attrition.json"), j(cfg, "d07_results.json", base="diag")
    TK = {"A->B": "AB", "B->C": "BC", "A->C": "AC", "C->D": "CD"}

    def pct1(key, v):
        R.add(key, v)
        R.fmt[key] = f"{100 * v:.1f}\\%"

    # datos: cobertura, minutos, exploración previa de 2024/25, Romeu con minutos R1
    m = pd.read_parquet(cfg.out("minutes_player_match.parquet"))
    cov = pd.read_csv(cfg.out("coverage_by_club_season.csv"))
    full = set(cov[(cov.season == "A") & (cov.matches_covered == 38)].team)
    nfull = m[(m.season == "A") & m.team.notna()].groupby("match_id").team.agg(lambda x: sum(t in full for t in set(x)))
    R.add("rob.covTwoFull", int((nfull == 2).sum()), "int")
    R.add("rob.covOneFull", int((nfull == 1).sum()), "int")
    mv = j(cfg, "minutes_validation.json")
    for s_ in ("A", "B"):
        pct1(f"rob.minGtTen{s_}", mv[s_]["share_abs_err_gt_flag"])
        dis = pd.read_csv(cfg.out(f"minutes_discrepancies_{s_}.csv"))
        R.add(f"rob.minMaxErr{s_}", float(dis.abs_err.max()) if len(dis) else 0.0, "int")
    th = d07["min_threshold_1000"]
    R.add("rob.prelimKOne", th["k1_2324_to_2425_naive_current_score"]["rho"])
    R.add("rob.prelimKTwo", th["k2_2223_to_2425_naive_current_score"]["rho"])
    ps = pd.read_parquet(cfg.out("players_season.parquet"))
    rom = ps[(ps.season == "A") & ps.player_name.str.contains("Romeu", na=False)]
    if len(rom):
        R.add("rob.romeuMinutesRone", float(rom.minutes_covered.iloc[0]), "int")
    # escala (R6)
    sc = rob["R6_scale"]
    R.add("rob.scaleEffSdC", sc["eff|C"]["sd"], "r1")
    R.add("rob.scaleEffMinC", sc["eff|C"]["min"], "int")
    R.add("rob.scaleEffMaxC", sc["eff|C"]["max"], "int")
    R.add("rob.scaleVolSdC", sc["vol|C"]["sd"], "r1")
    R.add("rob.halfWidthSd", sc["eff_interval"]["half_width_k1_sd"])
    def ordinal(key, v):  # percentil como ordinal en inglés: 14th, 90th, 21st...
        n = int(round(100 * v))
        suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        R.add(key, n, "int")
        R.fmt[key] = f"{n}{suf}"
    ordinal("rob.medPctLo", sc["eff_interval"]["median_player_pct_lo"])
    ordinal("rob.medPctHi", sc["eff_interval"]["median_player_pct_hi"])
    for key in ("B->C", "A->C"):
        for mod in ("M0", "M2"):
            R.add(f"rob.maeSd.{TK[key]}.{mod}", sc[f"eff_mae|{key}"][mod])
    # completado (R0)
    for k_, v in rob["R0_completed"]["mae_diff"].items():
        nm, key = k_.split("|")
        R.add_est(f"rob.maeDiff.{nm}.{TK[key]}", v, "r1")
    for s_, v in rob["R0_completed"]["corr_between"].items():
        R.add_est(f"rob.corr.{s_}", v)
    for key, a in mm["accuracy"].items():
        for mod in ("M0", "M2", "M3"):
            R.add_est(f"rob.topq.eff.{TK[key]}.{mod}", a[f"topq_{mod}"])
        R.add_est(f"rob.dM3.eff.{TK[key]}", a["d_rho_M3_W"], "r3")
    for key, a in rob["R0_completed"]["vol_accuracy"].items():
        for mod in ("M0", "M2"):
            R.add_est(f"rob.topq.vol.{TK[key]}.{mod}", a[f"topq_{mod}"])
            R.add(f"rob.mae.vol.{TK[key]}.{mod}", a[f"mae_{mod}"]["est"], "r1")
    # cobertura de intervalos (preespecificada) y Brier
    for key in ("B->C", "A->C"):
        u = unc["by_transition"][key]
        R.add(f"rob.covConf.{TK[key]}", u["coverage_conf"], "pct")
        R.add(f"rob.covConfLo.{TK[key]}", u["coverage_conf_ci95"][0], "pct")
        R.add(f"rob.covConfHi.{TK[key]}", u["coverage_conf_ci95"][1], "pct")
        for g in ("DEF", "MID", "FWD"):
            R.add(f"rob.covGauss.{TK[key]}.{g}", u["coverage_gauss_by_pos"][g], "pct")
    R.add("rob.calibTopBinN", rob["R7_calibration"]["B->C"][-1]["n"], "int")
    iv = j(cfg, "implied.json")["implied_vs_observed"]
    R.add("rob.impliedEvalKOne", iv["k1"]["implied_M2_W"]["est"], "r3")
    R.add("rob.impliedEvalKTwo", iv["k2"]["implied_M2_W"]["est"], "r3")
    # atrición, AUC y minutos ajustados por edad (preespecificados)
    for key in ("A->B", "B->C"):
        R.add(f"rob.aucAbsent.{TK[key]}", ext["presence_auc"][key]["n_absent"], "int")
        mn = ext["minutes_next"][key]
        R.add_est(f"rob.minPartialAge.{TK[key]}", {"est": mn["partial_age_W"], "ci95": mn["partial_age_W_ci95"]})
    # contexto del club (R1)
    for key, v in rob["R1_club_context"].items():
        t = TK[key]
        R.add(f"rob.stay.{t}.nStay", v["n_stayers"], "int")
        R.add(f"rob.stay.{t}.nMove", v["n_movers"], "int")
        for nm in ("eff", "vol"):
            for lab in ("stayers", "movers"):
                R.add_est(f"rob.stay.{t}.{nm}.{lab}", v[f"{nm}_{lab}"])
        for kk, vv in v["partial"].items():
            R.add_est(f"rob.partial.{t}.{kk}", vv)
    # valor de mercado con club (R2)
    for key, v in rob["R2_market_value_club"].items():
        for kk in ("eff_level_next_club", "vol_level_next_club", "diff_level_next_club",
                   "eff_level_now_club", "vol_level_now_club", "vol_change_club", "eff_change_club"):
            R.add_est(f"rob.mvclub.{TK[key]}.{kk.replace('_', '.')}", v[kk])
    # cuartil superior (R3)
    for k_, rows in rob["R3_top_quartile"].items():
        nm, key = k_.split("|")
        top = [r for r in rows if r["q_t"] == 4][0]
        R.add(f"rob.tq.{nm}.{TK[key]}.n", top["n"], "int")
        R.add_est(f"rob.tq.{nm}.{TK[key]}.obs", {"est": top["obs"], "ci95": top["obs_ci95"]})
        if "model" in top:
            R.add(f"rob.tq.{nm}.{TK[key]}.model", top["model"])
    # Δρ ponderado y bootstrap conjunto (R4, R5)
    for sec, lab in (("R4_ipw_delta", "ipw"), ("R5_joint_bootstrap", "joint")):
        for k_, v in rob[sec].items():
            nm, key = k_.split("|")
            R.add_est(f"rob.{lab}.{nm}.{TK[key]}", v, "r3")
    jt = [v["ci95"][1] for v in rob["R5_joint_bootstrap"].values()]
    R.add("rob.jointMaxHi", max(jt), "r3")
    # Wilson para la edad de quienes salen y quienes siguen (A→B)
    tab = {r["status"]: r for r in att["tables"]["A->B"]}

    def wil(k, n, z=1.959964):
        p_ = k / n
        den = 1 + z ** 2 / n
        c = (p_ + z ** 2 / (2 * n)) / den
        h = z * np.sqrt(p_ * (1 - p_) / n + z ** 2 / (4 * n ** 2)) / den
        return [c - h, c + h]
    R.add("rob.ageOldCut", 30, "int")  # corte de edad de la tabla de atrición (r09: age_ge30)
    for st_, lab in (("absent", "absent"), ("eligible", "eligible")):
        r = tab[st_]
        lo, hi = wil(round(r["age_ge30"] * r["n"]), r["n"])
        R.add(f"rob.age30.{lab}.lo", lo, "pct")
        R.add(f"rob.age30.{lab}.hi", hi, "pct")
        R.add(f"rob.age30.{lab}.n", r["n"], "int")

    # ---- tablas ----
    # Tabla 3: cuartil superior con la definición preregistrada, ambos resúmenes
    L = ["\\begin{tabular}{lccccc}", "\\toprule",
         " & \\multicolumn{3}{c}{Per-action efficiency} & \\multicolumn{2}{c}{Contribution per 90} \\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}",
         "Quartile this season & $n$ & Observed (95\\% CI) & Model & $n$ & Observed (95\\% CI) \\\\", "\\midrule"]
    qlab = {4: "Top", 3: "Third", 2: "Second", 1: "Bottom"}
    for key, hz in (("B->C", "One season ahead"), ("A->C", "Two seasons ahead")):
        L.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{hz}}}}} \\\\")
        e = {r["q_t"]: r for r in rob["R3_top_quartile"][f"eff|{key}"]}
        vv = {r["q_t"]: r for r in rob["R3_top_quartile"][f"vol|{key}"]}
        for q in (4, 3, 2, 1):
            L.append(f"{qlab[q]} & {e[q]['n']} & {num(e[q]['obs'])} [{num(e[q]['obs_ci95'][0])}, {num(e[q]['obs_ci95'][1])}] & "
                     f"{num(e[q]['model'])} & {vv[q]['n']} & {num(vv[q]['obs'])} [{num(vv[q]['obs_ci95'][0])}, "
                     f"{num(vv[q]['obs_ci95'][1])}] \\\\")
        L.append("\\addlinespace")
    L = L[:-1] + ["\\bottomrule", "\\end{tabular}"]
    _table(cfg, "tab_top_quartile_both.tex", L)
    # S4: contexto del club
    L = ["\\begin{tabular}{llcc}", "\\toprule", "Transition & Summary & Stayed at club & Changed club \\\\", "\\midrule"]
    for key, v in rob["R1_club_context"].items():
        for nm, lab in (("eff", "Per-action efficiency"), ("vol", "Contribution per 90")):
            L.append(f"{key.replace('->', ARROW) if nm == 'eff' else ''} & {lab} & "
                     f"{est_ci(v[nm + '_stayers'])} ($n$ = {v['n_stayers']}) & {est_ci(v[nm + '_movers'])} ($n$ = {v['n_movers']}) \\\\")
        L.append("\\addlinespace")
    L = L[:-1] + ["\\bottomrule", "\\end{tabular}"]
    _table(cfg, "tab_club_stay_move.tex", L)
    L = ["\\begin{tabular}{llcc}", "\\toprule", "Transition & Quantity & Adjusted for position & Position and clubs \\\\", "\\midrule"]
    for key, v in rob["R1_club_context"].items():
        pr = v["partial"]
        for i, (nm, lab) in enumerate((("eff", "Per-action efficiency"), ("vol", "Contribution per 90"),
                                       ("diff", "Difference (contribution $-$ efficiency)"))):
            L.append(f"{key.replace('->', ARROW) if i == 0 else ''} & {lab} & {est_ci(pr[nm + '_pos'])} & {est_ci(pr[nm + '_club'])} \\\\")
        L.append("\\addlinespace")
    L = L[:-1] + ["\\bottomrule", "\\end{tabular}"]
    _table(cfg, "tab_club_partial.tex", L)
    # S4: valor de mercado ajustado por club
    tlab = {"A->B": "2022/23 $\\to$ 2023/24", "B->C": "2023/24 $\\to$ 2024/25", "C->D": "2024/25 $\\to$ 2025/26"}
    L = ["\\begin{tabular}{lccc}", "\\toprule", "Outcome and transition & Per-action efficiency & Contribution per 90 & Difference \\\\",
         "\\midrule"]
    for kk, lab in (("level_next", "Market value, end of next season"), ("level_now", "Market value, end of same season"),
                    ("change", "Change in market value")):
        L.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{lab}}}}} \\\\")
        for key in ("A->B", "B->C", "C->D"):
            v = rob["R2_market_value_club"][key]
            L.append(f"{tlab[key]} & {est_ci(v['eff_' + kk + '_club'])} & {est_ci(v['vol_' + kk + '_club'])} & "
                     f"{est_ci(v['diff_' + kk + '_club'])} \\\\")
        L.append("\\addlinespace")
    L = L[:-1] + ["\\bottomrule", "\\end{tabular}"]
    _table(cfg, "tab_mv_club.tex", L)
    # S4: comparación de modelos bajo pesos IPW y bootstrap conjunto
    L = ["\\begin{tabular}{lccc}", "\\toprule", "Horizon & Main analysis & Attrition weights & Joint bootstrap \\\\", "\\midrule"]
    ad = j(cfg, "addendum1.json")
    for nm, lab in (("eff", "Per-action efficiency"), ("vol", "Contribution per 90")):
        L.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{lab}}}}} \\\\")
        for key, hz in (("B->C", "One season"), ("A->C", "Two seasons")):
            main_ = mm["accuracy"][key]["d_rho_M2_W"] if nm == "eff" else ad["E1"][key]["d_rho_M2_W"]
            L.append(f"{hz} & {est_ci(main_, 'r3')} & "
                     f"{est_ci(rob['R4_ipw_delta'][nm + '|' + key], 'r3')} & {est_ci(rob['R5_joint_bootstrap'][nm + '|' + key], 'r3')} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    _table(cfg, "tab_model_robust.tex", L)
    # S4: cobertura de los intervalos por grupo
    L = ["\\begin{tabular}{llcccc}", "\\toprule", "Horizon & Interval & All (95\\% CI) & Defenders & Midfielders & Forwards \\\\", "\\midrule"]
    for key, hz in (("B->C", "One season"), ("A->C", "Two seasons")):
        u = unc["by_transition"][key]
        for meth, lab in (("gauss", "Normal"), ("conf", "Conformal")):
            ci_ = u[f"coverage_{meth}_ci95"]
            by = u[f"coverage_{meth}_by_pos"]
            L.append(f"{hz if meth == 'gauss' else ''} & {lab} & {100 * u['coverage_' + meth]:.0f}\\% [{100 * ci_[0]:.0f}, {100 * ci_[1]:.0f}] & "
                     + " & ".join(f"{100 * by[g]:.0f}\\%" for g in ("DEF", "MID", "FWD")) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    _table(cfg, "tab_interval_coverage.tex", L)
    # S4: calibración con todos los bins
    L = ["\\begin{tabular}{llccc}", "\\toprule", "Horizon & Predicted probability & $n$ & Mean predicted & Observed (95\\% CI) \\\\", "\\midrule"]
    for key, hz in (("B->C", "One season"), ("A->C", "Two seasons")):
        for i, b in enumerate(rob["R7_calibration"][key]):
            lo_, hi_ = b["bin"].strip("(]").split(", ")
            lab = f"{max(0.0, float(lo_)):.2f} to {float(hi_):.2f}"
            L.append(f"{hz if i == 0 else ''} & {lab} & {b['n']} & {num(b['p_mean'])} & {num(b['obs_rate'])} "
                     f"[{num(b['obs_ci95'][0])}, {num(b['obs_ci95'][1])}] \\\\")
        L.append("\\addlinespace")
    L = L[:-1] + ["\\bottomrule", "\\end{tabular}"]
    _table(cfg, "tab_calibration.tex", L)


def extend(cfg: Config, R) -> None:
    design_numbers(cfg)
    extra_macros(cfg, R)
    tables_and_figure(cfg)
    supplement_tables(cfg)
    robustness(cfg, R)

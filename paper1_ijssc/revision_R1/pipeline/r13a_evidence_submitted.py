"""R13a — Evidencia sobre la versión ENVIADA para la respuesta a los revisores (REQ-A14, T2.9).

Toma las cifras de los diagnósticos d01–d07 (outputs/d0x_results.json, scripts de solo lectura sobre lo enviado) y
del hallazgo de minutos (outputs/r1/data_audit.json). Cada cifra lleva su archivo y su clave de origen.
Salida: results/evidence_submitted_version.md y results/evidence_submitted_version.json.
"""
from __future__ import annotations

import json

from common import R1, Config, load_config


def run(cfg: Config) -> dict:
    o = R1 / "outputs"
    d = {k: json.loads((o / f"{k}_results.json").read_text()) for k in ("d01", "d02", "d03", "d04", "d05", "d06", "d07")}
    audit = json.loads(cfg.out("data_audit.json").read_text())
    E: dict = {}
    L = ["# Evidencia sobre la versión enviada (para la respuesta a los revisores)", "",
         "Todas las cifras salen de diagnósticos de solo lectura sobre lo enviado (commit 62468f6). Formato: cifra — "
         "`archivo` › clave. No se usan en el manuscrito nuevo salvo como contraste explícito con lo enviado.", ""]

    def add(com, text, src):
        E.setdefault(com, []).append({"text": text, "source": src})
        L.append(f"- **{com}.** {text} — `{src}`")

    L.append("## Comentarios mayores (McHale)")
    r = d["d02"]
    add("M1", f"El ρk enviado se reproduce: {r['B_rho_operative']['pooled']} (agrupado).",
        "outputs/d02_results.json › B_rho_operative.pooled")
    add("M1", f"Sin términos de edad, ρk dentro de posición = {r['C_counterfactual_rho']['no_age_terms']['within_weighted']}.",
        "d02 › C_counterfactual_rho.no_age_terms.within_weighted")
    add("M1", f"Curvas de edad ×0,25 → ρ3 = {r['C_counterfactual_rho']['age_scale_0.25']['pooled'][2]}; "
              f"×2 → ρ3 = {r['C_counterfactual_rho']['age_scale_2.0']['pooled'][2]}.", "d02 › C_counterfactual_rho")
    add("M1", f"En 50 reajustes razonables, ρ3 ∈ {d['d04']['rho3_range_all_fits']} y ρ2 ∈ {d['d04']['rho2_range_all_fits']}.",
        "outputs/d04_results.json")
    add("M2", f"El score actual dentro de la muestra da ρ = {d['d03']['1_baselines_211']['alpha_0.06']['naive_current_score']}, "
              f"frente al OOF del operativo de {d['d03']['1_baselines_211']['alpha_0.06']['oof_operative']['mean']}.",
        "outputs/d03_results.json › 1_baselines_211.alpha_0.06")
    add("M2", f"Simulación condicionada al modelo: ρ(resultado simulado, actual) = "
              f"{r['D_predictive_simulation']['rho_simulated_outcome_vs_current_mean']}.",
        "d02 › D_predictive_simulation")
    ref = r["C_refits"]
    add("M3", "Picos de la función de edad (operativo): GK {f_GK}, DEF {f_DEF}, MID {f_MID}, FWD {f_FWD}. La trayectoria "
              "recursiva desde (20, 50) da GK {r_GK}, DEF {r_DEF}, MID {r_MID}, FWD {r_FWD}; desde (20, 30): {r30}; desde "
              "(20, 70): sin máximo interior (pico = 20).".format(
                  f_GK=ref["operative_published"]["age_function_peak_GK"], f_DEF=ref["operative_published"]["age_function_peak_DEF"],
                  f_MID=ref["operative_published"]["age_function_peak_MID"], f_FWD=ref["operative_published"]["age_function_peak_FWD"],
                  r_GK=ref["operative_published"]["recursive_peak_GK_start50"], r_DEF=ref["operative_published"]["recursive_peak_DEF_start50"],
                  r_MID=ref["operative_published"]["recursive_peak_MID_start50"], r_FWD=ref["operative_published"]["recursive_peak_FWD_start50"],
                  r30=[ref["operative_published"][f"recursive_peak_{p}_start30"] for p in ("GK", "DEF", "MID", "FWD")]),
        "d02 › C_refits.operative_published")
    add("M3", f"El pico recursivo de GK varía entre {min(d['d04']['rec_peak_GK_values'])} y {max(d['d04']['rec_peak_GK_values'])} "
              f"según el umbral y el estimador; solo {d['d04']['n_pass_filter']} de {d['d04']['n_fits']} reajustes pasan "
              "el filtro [24, 31].", "outputs/d04_results.json")
    add("M4", f"βs: ridge {ref['ridge_raw']['beta_s']}, ridge con βs sin penalizar {ref['ridge_beta_unpen']['beta_s']}, OLS "
              f"{ref['ols']['beta_s']}. ρk con βs sin penalizar = {ref['ridge_beta_unpen']['rho']['pooled'][:3]}; con OLS = "
              f"{ref['ols']['rho']['pooled'][:3]}.", "d02 › C_refits")
    add("M5", f"CV anidada (grilla reducida): ρ externo {d['d03']['2_nested_cv']['outer_rho_vs_alpha006_target_mean']} frente a "
              f"{d['d03']['2_nested_cv']['apparent_best_inner_rho_mean']} aparente.", "d03 › 2_nested_cv")
    add("M6", "Pesos re-estimados dentro del fold: ρ OOF {a} → {b}; persistencia del fold de prueba {c} → {e}; con pesos "
              "iguales la persistencia es {f}.".format(a=d["d05"]["foldwise_alpha0.06"]["oof_rho_fixed_weights_mean"],
                                                       b=d["d05"]["foldwise_alpha0.06"]["oof_rho_foldwise_weights_mean"],
                                                       c=d["d05"]["foldwise_alpha0.06"]["test_fold_y2y_fixed_mean"],
                                                       e=d["d05"]["foldwise_alpha0.06"]["test_fold_y2y_foldwise_mean"],
                                                       f=d["d05"]["rho_y2y_alpha0.0"]["equal_event_weights"]),
        "outputs/d05_results.json")
    mi = audit["minutes"]
    add("M7", f"Los minutos usados por P04 están inflados: razón mediana enviado/real {mi['psf_over_true_ratio_median']:.2f}; "
              f"≥ 1.000 min con minutos reales: A {mi['true_ge1000_AB']['A']} (enviado {mi['psf_ge1000_AB']['A']}), "
              f"B {mi['true_ge1000_AB']['B']} (enviado {mi['psf_ge1000_AB']['B']}); solo "
              f"{mi['submitted_211_pairs_true_ge1000_both']} de los 211 pares cumplen en ambas temporadas.",
        "outputs/r1/data_audit.json › minutes")
    add("M7", "Cobertura por club: {x}.".format(x=d["d01"]["minutes_partial_clubs"]), "d01 › minutes_partial_clubs")
    p6 = d["d06"]["2_persistence"]
    add("M8", f"Persistencia propia de PSR (versión del código) en los 211: {p6['code_scorePSR|training_211']['rho']} "
              f"{p6['code_scorePSR|training_211']['ci95']}; versión de la Ec. 2: {p6['eq2_pooled_percentile|training_211']['rho']} "
              f"{p6['eq2_pooled_percentile|training_211']['ci95']}.", "outputs/d06_results.json › 2_persistence")
    add("M8", f"PSR de la Ec. 2 frente a la proporción de pases: DEF {d['d06']['3_event_mix']['psr_pooled_vs_share_Pass_DEF']}, "
              f"MID {d['d06']['3_event_mix']['psr_pooled_vs_share_Pass_MID']}; Spearman de la Ec. 2 frente al código "
              f"{d['d06']['1_code_vs_eq2']['spearman_scorePSR_code_vs_pooled_percentile']}.", "d06 › 3_event_mix, 1_code_vs_eq2")
    add("M8", f"`under_pressure` por tipo (A/B/C): {audit['under_pressure_by_type']}.", "outputs/r1/data_audit.json")

    L.append("")
    L.append("## Comentarios menores (McHale)")
    add("m1", f"Enlace nombre→ID en lo enviado: {d['d01']['linkage']}; IDs con varios nombres: "
              f"{d['d01']['player_ids_with_multiple_event_names']}; duplicados: {d['d01']['p03_duplicate_player_season']}.",
        "d01 › linkage")
    prof = {p["player_name"]: (p.get("n_matches_ev"), p.get("minutes_played"), p.get("score_h0")) for p in d["d01"]["profile_players"]}
    add("m1", f"Exposición de perfiles de la Fig. 5 (partidos, minutos, score h0): Romeu {prof.get('Oriol Romeu Vidal')}, "
              f"Otto {prof.get('Jonathan Castro Otto')}, Nwaneri {prof.get('Ethan Chidiebere Nwaneri')}.", "d01 › profile_players")
    add("m2", f"ρk dentro de posición (enviado): DEF {r['B_rho_operative']['DEF']}, MID {r['B_rho_operative']['MID']}, "
              f"FWD {r['B_rho_operative']['FWD']}, GK {r['B_rho_operative']['GK']}.", "d02 › B_rho_operative")
    add("m6", f"MDD: fórmula con r23 supuesto 0,95 → {d['d03']['3_mdd']['paper_formula_r23_assumed_0.95']}; r23 empírico "
              f"{d['d03']['3_mdd']['r23_empirical_oof_predictions']} → {d['d03']['3_mdd']['paper_formula_r23_empirical']}; "
              f"bootstrap → {d['d03']['3_mdd']['mdd_from_bootstrap_se_(za+zb)*se']}.", "d03 › 3_mdd")
    add("m7", "Etiqueta de la Fig. 5 = inicial + último token del nombre: 'D. Martin', 'V. Dijk', 'O. Vidal', 'J. Otto'.",
        "d01 › profile_players.label_fig5_reproduced")
    add("m8", f"Convención de edad enviada: entera al 30 de junio de fin de temporada (coincidencia "
              f"{d['d01']['age_match_rate_by_reference_date']['2023/2024|season_end_06-30']:.3f}).",
        "d01 › age_match_rate_by_reference_date")
    add("m9", f"Cambios de grupo entre temporadas (enviado, grupo canónico único): "
              f"{d['d01']['position_group_change_between_seasons']}.", "d01 › position_group_change_between_seasons")
    add("m10", f"Universo de proyección enviado por minutos: {d['d01']['projection_universe_minutes']}; temporada base: "
               f"{d['d01']['projection_universe_base_season']}.", "d01")
    cfg.res("evidence_submitted_version.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    cfg.res("evidence_submitted_version.json").write_text(json.dumps(E, indent=2, ensure_ascii=False, default=str))
    return E


if __name__ == "__main__":
    run(load_config())
    print((load_config().res("evidence_submitted_version.md")).read_text()[:3000])

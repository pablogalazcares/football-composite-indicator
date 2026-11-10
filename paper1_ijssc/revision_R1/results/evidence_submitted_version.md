# Evidencia sobre la versión enviada (para la respuesta a los revisores)

Todas las cifras salen de diagnósticos de solo lectura sobre lo enviado (commit 62468f6). Formato: cifra — `archivo` › clave. No se usan en el manuscrito nuevo salvo como contraste explícito con lo enviado.

## Comentarios mayores (McHale)
- **M1.** El ρk enviado se reproduce: [0.9402, 0.7204, 0.4193, 0.2045, 0.0839] (agrupado). — `outputs/d02_results.json › B_rho_operative.pooled`
- **M1.** Sin términos de edad, ρk dentro de posición = [1.0, 1.0, 1.0, 1.0, 1.0]. — `d02 › C_counterfactual_rho.no_age_terms.within_weighted`
- **M1.** Curvas de edad ×0,25 → ρ3 = 0.8067; ×2 → ρ3 = 0.2527. — `d02 › C_counterfactual_rho`
- **M1.** En 50 reajustes razonables, ρ3 ∈ [0.2, 0.488] y ρ2 ∈ [0.493, 0.784]. — `outputs/d04_results.json`
- **M2.** El score actual dentro de la muestra da ρ = 0.4872, frente al OOF del operativo de 0.4613. — `outputs/d03_results.json › 1_baselines_211.alpha_0.06`
- **M2.** Simulación condicionada al modelo: ρ(resultado simulado, actual) = [0.56, 0.274, 0.126, 0.054, 0.021]. — `d02 › D_predictive_simulation`
- **M3.** Picos de la función de edad (operativo): GK 18.6, DEF 28.8, MID 22.7, FWD 21.7. La trayectoria recursiva desde (20, 50) da GK 24, DEF 31, MID 24, FWD 24; desde (20, 30): [25, 31, 26, 26]; desde (20, 70): sin máximo interior (pico = 20). — `d02 › C_refits.operative_published`
- **M3.** El pico recursivo de GK varía entre 22 y 34 según el umbral y el estimador; solo 3 de 50 reajustes pasan el filtro [24, 31]. — `outputs/d04_results.json`
- **M4.** βs: ridge 0.4755, ridge con βs sin penalizar 0.4759, OLS 0.491. ρk con βs sin penalizar = [0.9403, 0.7207, 0.4197]; con OLS = [0.8994, 0.6602, 0.3776]. — `d02 › C_refits`
- **M5.** CV anidada (grilla reducida): ρ externo 0.4303 frente a 0.4593 aparente. — `d03 › 2_nested_cv`
- **M6.** Pesos re-estimados dentro del fold: ρ OOF 0.4536 → 0.4143; persistencia del fold de prueba 0.4794 → 0.4397; con pesos iguales la persistencia es 0.327. — `outputs/d05_results.json`
- **M7.** Los minutos usados por P04 están inflados: razón mediana enviado/real 1.23; ≥ 1.000 min con minutos reales: A 270 (enviado 351), B 271 (enviado 337); solo 160 de los 211 pares cumplen en ambas temporadas. — `outputs/r1/data_audit.json › minutes`
- **M7.** Cobertura por club: {'n_player_seasons_at_20_match_clubs': 535, 'max_minutes_played_PSF': 2224.0, 'max_n_matches_PSF': 25.0, 'share_ge_1000_at_20_clubs': 0.5121495327102804, 'share_ge_1000_at_38_clubs': 0.7486437613019892}. — `d01 › minutes_partial_clubs`
- **M8.** Persistencia propia de PSR (versión del código) en los 211: 0.433 [0.307, 0.542]; versión de la Ec. 2: 0.649 [0.541, 0.736]. — `outputs/d06_results.json › 2_persistence`
- **M8.** PSR de la Ec. 2 frente a la proporción de pases: DEF 0.522, MID 0.525; Spearman de la Ec. 2 frente al código 0.824. — `d06 › 3_event_mix, 1_code_vs_eq2`
- **M8.** `under_pressure` por tipo (A/B/C): {'A': {'Carry': 0.322, 'Clearance': 1.0, 'Dribble': 1.0, 'Duel': 1.0, 'Pass': 0.155, 'Shot': 0.355}, 'B': {'Carry': 0.334, 'Clearance': 1.0, 'Dribble': 1.0, 'Duel': 1.0, 'Pass': 0.163, 'Shot': 0.299}, 'C': {'Carry': 0.367, 'Clearance': 1.0, 'Dribble': 1.0, 'Duel': 1.0, 'Pass': 0.183, 'Shot': 0.357}}. — `outputs/r1/data_audit.json`

## Comentarios menores (McHale)
- **m1.** Enlace nombre→ID en lo enviado: {'event_names': 760, 'exact': 759, 'jaccard': 0, 'unresolved': 1, 'unresolved_events': 51, 'events_total': 1168714}; IDs con varios nombres: 0; duplicados: 0. — `d01 › linkage`
- **m1.** Exposición de perfiles de la Fig. 5 (partidos, minutos, score h0): Romeu (1, 75.0, 79.5), Otto (1, 94.0, 94.3), Nwaneri (1, nan, 44.0). — `d01 › profile_players`
- **m2.** ρk dentro de posición (enviado): DEF [0.9962, 0.9734, 0.8808, 0.6932, 0.47], MID [0.9651, 0.7946, 0.5149, 0.2892, 0.1496], FWD [0.972, 0.8026, 0.5088, 0.2454, 0.074], GK [0.721, 0.2096, 0.0042, -0.0763, -0.08]. — `d02 › B_rho_operative`
- **m6.** MDD: fórmula con r23 supuesto 0,95 → 0.0457; r23 empírico 0.9921 → 0.0185; bootstrap → 0.0252. — `d03 › 3_mdd`
- **m7.** Etiqueta de la Fig. 5 = inicial + último token del nombre: 'D. Martin', 'V. Dijk', 'O. Vidal', 'J. Otto'. — `d01 › profile_players.label_fig5_reproduced`
- **m8.** Convención de edad enviada: entera al 30 de junio de fin de temporada (coincidencia 0.997). — `d01 › age_match_rate_by_reference_date`
- **m9.** Cambios de grupo entre temporadas (enviado, grupo canónico único): {'players_in_both_seasons': 333, 'changed_group': 30, 'changed_group_in_training_211': 8}. — `d01 › position_group_change_between_seasons`
- **m10.** Universo de proyección enviado por minutos: {'lt_90': 7, 'lt_450': 149, 'lt_1000': 313, 'ge_1000': 419, 'missing': 27}; temporada base: {'2023/2024': 557, '2022/2023': 202}. — `d01`

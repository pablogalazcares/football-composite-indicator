# Verificación de afirmaciones (T3.9, REQ-W3)

Fecha: 2026-11-09. Esta versión es posterior a la lectura adversaria simulada (`simulated_review_R1.md`) y a la adenda 2.
Cubre cada afirmación del abstract, la introducción y las conclusiones, más las afirmaciones nuevas de la discusión.
Las cifras se citan por su macro (`results/numbers.tex`) o por su clave JSON en `outputs/r1/`.

Estados:
- **S**: sostenida por los resultados de R1.
- **L**: sostenida por literatura verificada el 2026-11-09 (`00_fuentes/verificacion_bibliografica.md`).
- **Q**: sostenida con un matiz que el texto declara.
- **E**: exploratoria y rotulada como tal.

## Abstract

| # | Afirmación | Evidencia | Estado |
|---|---|---|---|
| A1 | Los clubes se comprometen por varias temporadas | Introducción ¶1 (mchale_holmes_2023, muller_2017, frost_2025, ribeiro_2025, hecksteden_2026) | L |
| A2 | OBV, jugadores de campo de la Premier con ≥ `\MinMinutes` minutos, tres temporadas | `ingest_manifest.json`; `config/r1.yaml › exposure` | S |
| A3 | El indicador preespecificado fue la eficiencia por acción | `PREREGISTRO.md` §1 (cd5dcdc) | S |
| A4 | Todo lo estimado se fijó en las dos primeras temporadas; la tercera se exploró antes, así que la evaluación no es ciega | `r06_models` (solo A→B); `PREREGISTRO.md` §0 (0,46 y 0,31, `\RRobPrelimKOne`, `\RRobPrelimKTwo`) | S |
| A5 | Persistencia 0,51 [0,39; 0,62] y 0,29 [0,11; 0,43] | `metrics_main.json › persistence.B->C / A->C .rho_obs_W` | S |
| A6 | Agregar edad no ordena mejor que el valor actual | `metrics_main.json › accuracy.*.d_rho_M2_W`; también con IPW y bootstrap conjunto (`robustness.json › R4, R5`). Límite superior máximo 0,047 | Q (el texto dice que no se descarta una mejora pequeña) |
| A7 | Intervalos del 80 % cubrieron 79 % y 78 % | `uncertainty.json › coverage_gauss` | S |
| A8 | 0,44 de los del cuartil superior siguieron arriba | `robustness.json › R3_top_quartile.eff|B->C` (igual a `topq_M0`) | S |
| A9 | Contribución por 90: persistencia similar a k = 1 (0,50) y mayor a k = 2 (0,49; diferencia 0,21 [0,02; 0,39]) | `addendum1.json › E1, E3` | E |
| A10 | Asociada más fuertemente con el valor de mercado, ventaja que desaparece dentro de clubes | `addendum1.json › E2`; `robustness.json › R2` (diferencias con club 0,06, 0,00 y 0,03, todas con IC que incluye 0) | E |
| A11 | Ninguno predijo los minutos de la temporada siguiente | `addendum1.json › E3.minutes_next_*`; `external.json › minutes_next.*.partial_age_W` | S |
| A12 | La proyección supera ampliamente a la persistencia observada y refleja el modelo | `implied.json › implied_vs_observed` (0,996 y 0,952 frente a 0,51 y 0,29); contrafactuales de S5 | S |
| A13 | Conclusión: informativo pero impreciso a una temporada; a dos depende del resumen; horizontes mayores no evaluables | A5, A8, A9 y diseño | Q |

## Introducción

| # | Afirmación | Evidencia | Estado |
|---|---|---|---|
| I1 | Los datos de desempeño se usan para valorar jugadores (fees, valor de mercado) | mchale_holmes_2023, muller_2017 | L |
| I2 | Y para apoyar el reclutamiento junto al scouting | frost_2025, ribeiro_2025, hecksteden_2026 | L |
| I3 | Definición de indicador de desempeño | hughes_bartlett_2002 | L |
| I4 | La estabilidad es necesaria pero no suficiente | Argumento; motiva las asociaciones externas | S |
| I5 | Los event data son producto del análisis notacional: operadores codifican video; incluyen eventos sin balón como presiones | hughes_franks_2004, pappalardo_2019, statsbomb_spec_2019 | L |
| I6 | Link et al.: dangerousity desde zona, control, presión y densidad; valor de acción = cambio en dangerousity | link_2016 (verificado; el propio paper define los Action Values) | L |
| I7 | Lucey et al.: probabilidad de gol del tiro con rasgos estratégicos de tracking | lucey_2015 | L |
| I8 | VAEP y OBV usan event data; cambio estimado en probabilidades de marcar y recibir | decroos_2019, statsbomb_obv_2021 | L |
| I9 | Criterios de estabilidad de métricas; variabilidad partido a partido en la Premier | franks_2016, bush_2015 | L |
| I10 | Estabilidad de valores de acción evaluada dentro de temporada (dos mitades) | vanroy_2020 (mitades, Premier 2018/19); baron_2024 como métrica de habilidad relacionada | L |
| I11 | Estudios de pronóstico a una temporada o sobre valor de mercado | vanarem_2025 ("one year ahead"); baouan_2025 (valor de mercado a dos años) | L |
| I12 | Sistemas de béisbol: desempeño reciente, regresión a la media, edad | tango_marcel_2004, silver_pecota_2004 | L |
| I13 | Tres resultados primarios preespecificados y evaluación no ciega | `PREREGISTRO.md` §0 y §4 | S |
| I14 | La contribución se agregó después de que la eficiencia no mostró asociaciones | `PREREGISTRO_ADENDA_1.md` §0 | S |

## Conclusiones

| # | Afirmación | Evidencia | Estado |
|---|---|---|---|
| C1 | La eficiencia ordena moderadamente a una temporada y débilmente a dos | A5 | S |
| C2 | Los modelos con edad no agregan nada en este horizonte | A6 | Q |
| C3 | El acuerdo proyección–actual no mide el desempeño futuro | A12 | S |
| C4 | Resumir por 90 minutos conserva más orden a dos temporadas, pero su vínculo con el valor de mercado refleja diferencias entre clubes | A9, A10 | E |
| C5 | Útil a una temporada y, con más cautela, a dos | A5, A8, A9 | Q |

## Discusión y resultados: afirmaciones agregadas en esta pasada

| # | Afirmación | Evidencia | Estado |
|---|---|---|---|
| D1 | La contribución mantuvo su orden entre quienes cambiaron de club | `robustness.json › R1_club_context.A->C.vol_movers` = 0,60 [0,30; 0,75], n = 35 | E |
| D2 | El ajuste por club casi no cambia la diferencia entre resúmenes, pero su IC incluye cero | `R1 … partial.diff_club` 0,16 [−0,04; 0,42]; `diff_pos` 0,15 [−0,02; 0,32] | E |
| D3 | La persistencia de la eficiencia está en parte incorporada por los pesos | `sensitivity.json › equal_weights` (0,36 frente a 0,51) | S |
| D4 | Las edades ≥ 30 salen más entre 2022/23 y 2023/24 | `attrition.json` (31 % frente a 18 %, con IC de Wilson); en B→C no (15 % frente a 17 %) | S |
| D5 | Foden: puesto 48 de 67 en eficiencia y 25 en contribución (2023/24) | `transitions.parquet`; macros `foden.*` | S |
| D6 | OBV valora acciones de arquero; tipo y versión no publicados | statsbomb_obv_2021 (hudl.com, consultado el 2026-11-09) | L |
| D7 | Los clubes parciales solo se observan contra los diez completos | `minutes_player_match.parquet`: 90 partidos entre completos, 200 entre completo y parcial, 0 entre parciales | S |

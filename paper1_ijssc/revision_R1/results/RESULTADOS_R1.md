# Resultados R1 (insumo para HC-3)

Fecha: 2026-11-08.
- **Datos:** capa R1 con minutos reales, IDs de StatsBomb, edad al 1 de enero y posición por temporada.
- **Indicador:** OBV dentro de posición con pesos AR1 de A→B.
- **Diseño:** el preregistrado (`PREREGISTRO.md`, commit `cd5dcdc`).
- **Inferencia:** bootstrap de jugadores estratificado por posición (B = 2.000); IC 95 %.
- **Notación:** ρ_W = Spearman dentro de posición, promedio ponderado sobre DEF, MID y FWD. A = 2022/23, B = 2023/24,
  C = 2024/25.
- **Fuentes:** todas las cifras están en `results/paper_numbers.json` (macros en `results/numbers.tex`) y en
  `outputs/r1/*.json`.

> Las cifras preliminares de `01_diagnostico_ejecutivo.md` (0,46 y 0,31) venían del indicador compuesto y de minutos
> proxy. Las de este documento las reemplazan.

## Resumen en siete puntos

1. **Persistencia observada.**
   - Una temporada (B→C): ρ_W = **0,51** [0,39; 0,62]. Dos temporadas (A→C): ρ_W = **0,29** [0,11; 0,43].
   - Con los mismos 108 jugadores elegibles en las tres temporadas: 0,51 y 0,26.
2. **Por posición, a dos temporadas,** solo los delanteros conservan un orden distinguible de cero: FWD 0,51
   [0,16; 0,75]; DEF 0,20 [−0,06; 0,46]; MID 0,24 [−0,04; 0,49].
3. **El score actual es tan buen predictor como cualquier modelo preespecificado.**
   - Δρ_W del modelo con edad (M2) frente al score actual: −0,004 [−0,021; 0,010] a una temporada y −0,033
     [−0,098; 0,024] a dos.
   - Los términos de edad, estimados con 150 pares, son nulos en la práctica: edad −0,03 [−0,47; 0,32]; edad² −0,02
     [−0,09; 0,09].
4. **El horizonte implícito de un modelo determinista no mide predictibilidad.**
   - M2 proyectado se correlaciona 0,99 y 0,95 con el score actual a k = 1 y k = 2, cuando lo observado es 0,51 y 0,29.
   - El modelo enviado daba 0,94 y 0,72 agrupado (0,96 y 0,81 dentro de posición).
5. **Incertidumbre.**
   - El intervalo de 80 % mide ±10,5 puntos de percentil y su cobertura es buena: 79 % [73 %; 86 %] a k = 1 y 78 % a
     k = 2.
   - Un jugador del cuartil superior sigue ahí al año siguiente con probabilidad 0,50 (modelo: 0,49), y a dos años con
     0,34 (modelo: 0,37).
6. **Atrición y cobertura** mueven poco las cifras.
   - IPW: 0,50 a k = 1 y 0,33 a k = 2.
   - Clubes con cobertura parcial y completa dan valores similares.
   - Los mayores salen más: retención de 50 % (30–32 años) y 48 % (33 o más) en A→B, frente a 68 % (23 o menos).
7. **Validez externa nula (activa R-b).** Entre los elegibles, el score de una temporada no se asocia con:
   - la proporción de minutos de la siguiente: ρ_W 0,04 [−0,10; 0,17] en A→B y 0,09 [−0,05; 0,22] en B→C;
   - la permanencia en la liga: AUC 0,48 y 0,51;
   - el valor de mercado (Spearman parcial 0,09 [−0,05; 0,22]) ni su cambio (−0,06 [−0,19; 0,09]).

## 1. Muestras

| | A→B (entrenamiento) | B→C (k = 1) | A→C (k = 2) |
|---|---|---|---|
| Elegibles en ambas (todas las posiciones) | 161 | 183 | 147 |
| Jugadores de campo elegibles en ambas | 150 | 168 | 134 |

- Elegibles por temporada: 270 / 274 / 321.
- Elegibles en las tres temporadas: 117 (108 de campo).
- Minutos validados contra lineups: error mediano de 0,00 min.
- Hallazgo sobre lo enviado: los minutos de P04 estaban inflados (razón mediana 1,23). Solo 160 de los 211 pares
  cumplían ≥ 1.000 minutos reales en ambas temporadas (ver `data_audit_R1.md`).

## 2. P1, persistencia observada (`tables/tab_persistence.tex`, `figures/fig_persistence_by_position.pdf`)

| Transición | n | DEF | MID | FWD | ρ_W |
|---|---|---|---|---|---|
| A→B (in-sample para los pesos) | 150 | 0,43 [0,20; 0,61] | 0,41 [0,11; 0,64] | 0,66 [0,40; 0,84] | 0,48 [0,32; 0,60] |
| **B→C (k = 1)** | 168 | 0,44 [0,22; 0,61] | 0,55 [0,31; 0,72] | 0,59 [0,37; 0,75] | **0,51 [0,39; 0,62]** |
| **A→C (k = 2)** | 134 | 0,20 [−0,06; 0,46] | 0,24 [−0,04; 0,49] | 0,51 [0,16; 0,75] | **0,29 [0,11; 0,43]** |

- Sobre el conjunto común (108 jugadores): 0,51 (A→B), 0,51 (B→C) y 0,26 (A→C).
- La caída de k = 1 a k = 2 (0,51 → 0,29) es parecida a la de un AR(1) (0,51² ≈ 0,26), con IC anchos.

## 3. P2, precisión fuera de tiempo (`tables/tab_accuracy.tex`)

| Horizonte | Predictor | ρ_W | Δρ_W frente al score actual | MAE | Precisión en el cuartil superior |
|---|---|---|---|---|---|
| 1 temporada | M0, score actual | 0,51 [0,39; 0,62] | — | 7,2 | 0,44 |
| | M1, AR(1) + posición | 0,51 | 0 (equivalente en rango, a priori) | 6,3 | 0,44 |
| | **M2, + edad** | 0,51 [0,38; 0,62] | **−0,004 [−0,021; 0,010]** | 6,3 | 0,49 |
| | M3, + posición × edad | 0,51 | −0,009 [−0,033; 0,013] | 6,3 | 0,47 |
| 2 temporadas | M0 | 0,29 [0,11; 0,43] | — | 8,4 | 0,35 |
| | **M2** | 0,25 [0,08; 0,41] | **−0,033 [−0,098; 0,024]** | 7,5 | 0,29 |
| | M3 | 0,27 | −0,019 [−0,100; 0,053] | 7,5 | 0,32 |

M1 y M2 bajan el MAE porque regresan a la media, pero no mejoran el orden.

Modelo M2, ajustado con 150 pares de A→B:
- βs = 0,57 [0,41; 0,72];
- MID −3,3 [−5,5; −1,0] y FWD −1,4 [−4,7; 2,1] frente a DEF;
- efectos de edad nulos (§Resumen, punto 3; `figures/fig_age_effect_M2.pdf`).

## 4. P3, incertidumbre (`tables/tab_rank_uncertainty.tex`, `figures/fig_reliability_top_quartile.pdf`)

- σ OOF = 8,2. Intervalo gaussiano de 80 % = ±10,5 puntos; conformal = ±9,9.
- Cobertura gaussiana: 79 % [73; 86] a k = 1 (DEF 76 %, MID 90 %, FWD 71 %) y 78 % [72; 85] a k = 2. Conformal: 77 %
  y 76 %.
- Brier skill de P(cuartil superior) frente a la tasa base: 0,14 a k = 1 y 0,03 a k = 2. La curva de confiabilidad
  queda cerca de la diagonal.
- P(cuartil superior en t + k), observada frente a modelada:

| Cuartil en t | k = 1 | k = 2 |
|---|---|---|
| Superior | 0,50 / 0,49 | 0,34 / 0,37 |
| Tercero | 0,33 / 0,29 | 0,29 / 0,29 |
| Segundo | 0,13 / 0,17 | 0,33 / 0,23 |
| Inferior | 0,05 / 0,06 | 0,03 / 0,16 |

## 5. P4, atrición y cobertura (`tables/tab_attrition.tex`)

- Entre los elegibles en t que salen, la proporción de mayores de 30 es más alta en A→B (31 % de los ausentes frente a
  18 % de los retenidos). En B→C no se observa (15 % frente a 17 %).
- En las transiciones a un año, los ausentes vienen más de clubes con cobertura parcial: 52 % en A→B y 65 % en B→C,
  frente a 29 % y 36 % de los retenidos. A dos años no hay diferencia (36 % frente a 39 %).
- IPW (ρ_W):

| Transición | Sin ponderar | IPW [IC] |
|---|---|---|
| B→C | 0,51 | 0,50 [0,36; 0,61] |
| A→C | 0,29 | 0,33 [0,14; 0,47] |
| A→B | 0,48 | 0,45 [0,28; 0,59] |

- Por cobertura del club de origen (20 frente a 38 partidos):

| Transición | Cobertura parcial | Cobertura completa |
|---|---|---|
| A→B | 0,35 [0,03; 0,62] | 0,51 [0,33; 0,66] |
| B→C | 0,46 [0,22; 0,64] | 0,49 [0,32; 0,62] |
| A→C | 0,28 [−0,02; 0,51] | 0,24 [0,01; 0,45] |

## 6. P5, validez externa (`tables/tab_external.tex`)

Ver el resumen, punto 7. El IC de la asociación con los minutos de t+1 incluye 0 en ambas transiciones, y eso
**activa R-b**.

Lectura posible, que hay que decidir en HC-3 y **no se actuó sobre ella**:
- El indicador promedia OBV **por acción** (eficiencia) y percentiliza dentro de cada tipo de evento: no mide volumen
  ni contribución total.
- Entre jugadores que ya son regulares (≥ 1.000 min), esa eficiencia no se traduce en más minutos ni en más valor de
  mercado.
- Esto se alinea con la duda de McHale (M9): ¿calidad o estilo de juego?

## 7. P6, horizonte implícito frente al observado (`figures/fig_implied_vs_observed.pdf`)

- **ρk enviado, reproducido:**
  - agrupado: 0,9402 / 0,7204 / 0,4193 / 0,2045 / 0,0839;
  - dentro de posición: 0,96 / 0,81 / 0,60 / 0,38 / 0,22.
- **Sin términos de edad,** el ρk dentro de posición es 1 en todos los horizontes.
- **M2 (preregistrado), proyectado desde 2024/25:** 0,99 / 0,95 / 0,83 / 0,65 / 0,46. Con efectos de edad casi nulos,
  el "horizonte" se alarga artificialmente; con curvas empinadas (el envío) se acorta. **En ningún caso dice cuánto se
  puede predecir.**
- **Observado frente a implícito:** 0,51 frente a 0,99 a k = 1, y 0,29 frente a 0,95 a k = 2.

## 8. P8, sensibilidades (`tables/tab_sensitivity.tex`, `figures/fig_sensitivity_persistence.pdf`)

ρ_W observado a k = 1 y k = 2, y Δρ_W de M2:

| Variante | k = 1 | k = 2 | Δρ_W (M2) a k = 1 |
|---|---|---|---|
| Principal | 0,51 | 0,29 | −0,004 |
| Pesos iguales | 0,36 | 0,22 | 0,000 |
| Macro 0,5 / 0,5 | 0,50 | 0,27 | −0,046 [−0,084; −0,011] |
| Posición canónica | 0,52 | 0,29 | −0,006 |
| ≥ 500 min | 0,45 | 0,25 | −0,007 |
| ≥ 750 min | 0,49 | 0,26 | −0,027 [−0,050; −0,005] |
| ≥ 30 % de los minutos del club | 0,52 | 0,31 | −0,016 |
| Con arqueros | 0,48 | 0,29 | −0,004 |
| Mismo grupo en t y t+k | 0,52 | 0,29 | −0,003 |
| 1.000–1.999 min en t | 0,48 | 0,18 | −0,020 |
| ≥ 2.000 min en t | 0,55 | 0,39 | −0,008 |

- **Conclusión robusta:** persistencia de ~0,45–0,55 a un año y ~0,2–0,4 a dos.
- El modelo con edad nunca mejora al score actual, y en dos variantes empeora de forma significativa.
- Los pesos AR1, estimados solo con A→B, sí suben la persistencia fuera de tiempo (0,51 frente a 0,36 con pesos
  iguales).
- **Legacy frente a revisión (R-c):** en A→B, 0,495 frente a 0,481. En B→C, con la posición canónica como aproximación
  del legacy, 0,52 frente a 0,51.

## 9. Desviaciones del preregistro

No hubo cambios de diseño después de ver resultados. Precisiones de implementación:
1. **R-c.** El indicador legacy solo existe en A/B (C no forma parte del universo enviado). Se evaluó en A→B, y en
   B→C con la posición canónica como aproximación. El preregistro no fijaba la transición.
2. **`r05_samples`.** Se agregó `score_obv_legacy` a las transiciones para poder evaluar R-c. No cambia ninguna otra
   cifra.
3. **P3.** La confiabilidad se dibuja solo con bins de n ≥ 5; la tabla los tiene todos.

## 10. Criterios de G3

| Criterio | Resultado | ¿Se activa? |
|---|---|---|
| R-a: IC de ρ a k = 2 ≤ 0 en DEF, MID y FWD, y Δρ a k = 1 que incluye 0 | DEF y MID sí (cotas −0,06 y −0,04); FWD no (0,16). Δρ a k = 1 incluye 0 | **No** (falla la primera condición por FWD), pero queda cerca |
| R-b: IC de ρ(s_t, minutos en t+1) que incluye 0 en ambas transiciones | [−0,10; 0,17] y [−0,05; 0,22] | **Sí** |
| R-c: el ρ principal a k = 1 cambia > 0,10 entre legacy y revisión | 0,495 frente a 0,481 | No |
| R-d: cobertura a k = 1 a más de 10 puntos del 80 % | 79 % | No |

## 11. Qué puede afirmar el paper con esta evidencia (para decidir en HC-3)

**Sostenible:**
1. El orden dentro de posición de un indicador de valor por acción (OBV) persiste de forma moderada a una temporada
   (~0,5) y débil a dos (~0,3). A dos temporadas solo los delanteros conservan un orden distinguible de cero.
2. Un modelo con edad y posición no mejora el ranking frente a usar el score de la temporada actual.
3. Los horizontes implícitos de una proyección determinista (correlación proyección–actual) sobrestiman la
   predictibilidad por un factor de 2 a 3, y su forma depende de las curvas de edad elegidas. Es una advertencia útil
   para clubes y analistas.
4. La incertidumbre es grande y está bien calibrada: un jugador del cuartil superior tiene ~50 % de seguir ahí al año
   siguiente y ~1/3 a dos años.

**No sostenible:** que el indicador mida "calidad" en un sentido que los clubes valoren. Con la evidencia actual no
predice minutos, permanencia ni valor de mercado (R-b).

## 12. Decisiones para HC-3

1. **Licencia de 2024/25 (Hudl).** Sin ella se activa el fallback A + C (solo A→B y el apéndice algebraico).
2. **R-b, cómo encuadrar la validez externa.** Opciones:
   - (a) Reportarla como límite de constructo y titular el paper sobre persistencia y predictibilidad de la
     "eficiencia por acción", no de la calidad.
   - (b) Preregistrar un análisis adicional con una versión de volumen (OBV total por 90 minutos) como contraste.
     Sería un análisis nuevo, declarado como posterior al resultado R-b.
   - (c) Ambas.
3. **Título**, según 1 y 2 (SPEC §9.3).
4. **PSR:** apéndice o eliminación total.
5. **Valor de mercado de 2024/25** (scraping de Transfermarkt) y pedido de 2025/26: ¿se hacen?
6. **Formato:** con R-a cerca de activarse y R-b activo, ¿IJSSC Original Research o un formato más corto?

## 13. Estado de la Fase 3

Bloqueada hasta que HC-3 se apruebe en `DECISIONS.md`.

El borrador de respuesta (`05_respuesta_revisores_borrador.md`) y el diagnóstico (`01_…`) todavía muestran cifras
preliminares (0,46 / 0,42 / 0,31 / 0,25). La Fase 3 las reemplaza por las macros de `numbers.tex`.

---

## Adenda 1: contraste de volumen y valor de mercado ampliado (`PREREGISTRO_ADENDA_1.md`, commit `c431da2`)

Análisis posterior al resultado R-b, declarado así (HC3-2).
- **Indicadores:**
  - **volumen** = OBV total por 90 minutos, percentil dentro de (temporada, posición);
  - **eficiencia** = el indicador principal.
- **Enlace StatsBomb ↔ Transfermarkt:** 917 de 929 jugadores (98,7 %; elegibles ≈ 100 %). La muestra manual de 30 pares
  sale toda correcta (`results/tm_link_review.md`).
- **Valor de mercado:** valoración ≤ 30/06 de cada fin de temporada, desde el historial de Transfermarkt ya descargado.
- **Correlación entre indicadores, dentro de la misma temporada (ρ_W):** 0,38 (A), 0,48 (B) y 0,56 (C). Miden cosas
  relacionadas pero distintas.

### E1 y E3: persistencia (ρ_W, elegibles en ambas temporadas)

| Horizonte | Eficiencia | Volumen | Volumen − eficiencia (pareado) |
|---|---|---|---|
| 1 temporada (B→C, n = 168) | 0,51 [0,39; 0,62] | 0,50 [0,37; 0,61] | −0,02 [−0,17; 0,14] |
| 2 temporadas (A→C, n = 134) | 0,29 [0,11; 0,43] | **0,49 [0,34; 0,60]** | **+0,21 [0,02; 0,39]** |

- **Volumen a dos temporadas, por posición:** DEF 0,46 [0,24; 0,62], MID 0,51 [0,26; 0,71], FWD 0,52 [0,21; 0,76].
  Las tres quedan sobre 0.
- **Modelo con edad frente al score actual (volumen):** −0,034 [−0,072; 0,002] a una temporada y −0,083 [−0,167;
  0,004] a dos. Tampoco mejora.

### E1 y E3: minutos y permanencia en t+1

| | A→B | B→C |
|---|---|---|
| ρ_W con la proporción de minutos en t+1, volumen | 0,06 [−0,08; 0,19] | 0,02 [−0,12; 0,16] |
| Volumen − eficiencia | 0,02 [−0,14; 0,17] | −0,07 [−0,21; 0,08] |
| AUC de permanencia, volumen | 0,52 | 0,48 |

Ningún indicador predice los minutos ni la permanencia entre los jugadores regulares.

### E2 y E3: valor de mercado (Spearman parcial por edad, edad² y posición)

| Transición (n) | Eficiencia → VM t+1 | Volumen → VM t+1 | Volumen − eficiencia | Volumen → VM t (mismo año) | Volumen → Δ VM |
|---|---|---|---|---|---|
| A→B (246) | 0,04 [−0,09; 0,16] | **0,23 [0,11; 0,35]** | **0,19 [0,07; 0,33]** | 0,28 [0,17; 0,40] | −0,02 [−0,14; 0,10] |
| B→C (248) | 0,04 [−0,08; 0,16] | **0,25 [0,13; 0,37]** | **0,21 [0,09; 0,32]** | 0,26 [0,15; 0,38] | −0,01 [−0,14; 0,13] |
| C→D′ (296) | 0,19 [0,08; 0,29] | **0,30 [0,19; 0,40]** | **0,11 [0,01; 0,21]** | 0,27 [0,16; 0,37] | 0,15 [0,03; 0,25] |

**Lectura.** El volumen se alinea con lo que el mercado valora, pero sobre todo de forma contemporánea: anticipa el
cambio de valor solo en C→D′. La eficiencia se alinea poco (solo en C→D′, 0,19).

### Regla de recomendación preregistrada (§4 de la adenda)

- El volumen muestra validez externa: el parcial con VM t+1 está sobre 0 en las tres transiciones.
- Supera a la eficiencia: el contraste pareado está sobre 0 en las tres.
- **Recomendación de la regla:** reportar ambos indicadores y encuadrar el paper en el contraste **eficiencia por
  acción frente a contribución total**, con la persistencia como eje.
- La decisión final es de los autores (HC3-2).

### Implicaciones para el encuadre, si se adopta la recomendación

1. **Pregunta:** ¿cuánto se mantiene el ranking de un jugador, según OBV, una y dos temporadas después, y depende de
   cómo se resume OBV?
2. **Hallazgos:**
   - A una temporada ambos resúmenes persisten ~0,5.
   - A dos, la contribución total (~0,5) persiste más que la eficiencia por acción (~0,3).
   - Ningún modelo con edad mejora al score actual.
   - Los horizontes implícitos de una proyección determinista sobrestiman la predictibilidad.
   - La contribución se alinea con el valor de mercado (contemporáneo); ninguno predice minutos futuros.
3. **McHale M9 (¿calidad o estilo?).** Respuesta empírica: la eficiencia por acción se comporta más como un rasgo de
   estilo (persiste a un año, no se alinea con el mercado); la contribución total, más como una medida de rendimiento
   que el mercado reconoce.

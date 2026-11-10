# Preregistro del análisis R1 (G2)

Fecha: 2026-11-08. Se commitea **antes** de calcular cualquier métrica entre temporadas con la capa de datos R1, sobre
B→C o A→C. Lo que se cambie después de ver resultados se registra como desviación en `results/RESULTADOS_R1.md`, con
su motivo.

## 0. Declaración de procedencia (D-15)

La temporada 2024/25 no es una prueba ciega.
- En junio de 2026 (`paper1_3seasons/01_hallazgos_iniciales.md`) se calcularon con ella correlaciones observadas y
  precisión del modelo.
- El 08-11-2026 se calcularon también en el diagnóstico `d07`, con el indicador enviado y minutos proxy (k = 1 ≈ 0,46;
  k = 2 ≈ 0,31).

Lo que este preregistro garantiza es otra cosa: **ninguna decisión de diseño de abajo se elige mirando 2024/25**.
- Todas vienen de la spec (SPEC_R1 §3), escrita antes de esta fase.
- Todo parámetro estimado viene de la transición 2022/23→2023/24.

## 1. Datos e indicador (fijados en la Fase 1)

- **Temporadas:** A = 2022/23, B = 2023/24, C = 2024/25.
- **Indicador principal:** `score_obv_ar1`, en modo revisión.
  - Percentiles de OBV por tipo de evento dentro de (temporada, grupo de la temporada, tipo).
  - Pesos AR1 de P03, estimados con A→B; macro FOR/AGAINST de P03; shrinkage de Shot con K = 10.
- **Elegibilidad (D-04):** ≥ 1.000 minutos reales en partidos cubiertos en **cada** temporada involucrada, y score no
  nulo.
- **Posiciones:** DEF, MID y FWD (D-03).
  - La estratificación usa el grupo de la temporada de **origen** t.
  - El percentil del objetivo se calcula en su propia temporada y grupo.
  - Arqueros: solo descriptivo.
- **Edad (D-05):** decimal al 1 de enero de la temporada t.

## 2. Preguntas y métricas

**Notación:**
- s_t: score en la temporada t.
- ρ_W: Spearman dentro de posición, promedio ponderado por n sobre DEF, MID y FWD.
- ρ_P: Spearman agrupado (solo referencia).

### P1. Persistencia observada (REQ-A1)

- **Métrica:** ρ_W(s_t, s_{t+k}) sobre los elegibles en t y en t+k.
- **Transiciones:**
  - k = 1 en A→B (in-sample para los pesos: se reporta, rotulado así) y en **B→C (principal)**;
  - k = 2 en **A→C (principal)**.
- **Muestras:** (i) todos los elegibles en ambas temporadas; (ii) el conjunto común de elegibles en A, B y C.
- Se reportan ρ por posición y ρ_P.

### P2. Precisión fuera de tiempo (REQ-A2)

- **Modelos (D-08)**, ajustados por OLS en los pares de campo elegibles A→B:
  - M0: ŝ = s_t;
  - M1: AR(1) + posición;
  - M2: M1 + (edad − 26) + (edad − 26)²;
  - M3: M2 + posición × (edad, edad²), exploratorio.
- **Predicción:**
  - k = 1: desde B, se evalúa en C;
  - k = 2: proyección recursiva de dos pasos desde A, con la edad + 1 en el segundo paso; se evalúa en C.
- **Métricas:**
  - ρ_W(ŝ, s_{t+k}) (principal);
  - ρ_P;
  - MAE en puntos de percentil;
  - precisión en el cuartil superior: entre los predichos en el cuartil superior de su posición, la proporción
    observada en el cuartil superior. Los cuartiles se calculan entre los elegibles en ambas temporadas, por
    posición.
- **Comparación (D-10):** Δρ_W = ρ_W(Mx) − ρ_W(M0), con el mismo remuestreo. **La comparación principal es M2 frente a
  M0, a k = 1 y a k = 2.**
- **Equivalencia a priori:** dentro de una posición, M1 es una transformación monótona de s_t. Su ρ dentro de posición
  es idéntico al de M0, así que M1 solo se compara en ρ_P y en MAE.

### P3. Incertidumbre (REQ-A4, A5; D-11)

- **Intervalos de 80 % para M2, de dos tipos:**
  - gaussiano, ŝ ± z₀.₉ · σ, con σ igual a la DE de los residuos OOF (5 folds, semilla fija) en A→B;
  - conformal, ŝ ± q, con q el cuantil ⌈(n + 1) · 0,8⌉/n de los |residuos OOF| en A→B.
- **k = 2:** el ancho se multiplica por √(1 + β̂²) (propagación del AR(1); aproximación declarada).
- **Cobertura empírica** en B→C (k = 1) y A→C (k = 2), total y por posición.
- **P(cuartil superior en t + k):**
  - se calcula para cada jugador elegible en t por simulación (2.000 sorteos gaussianos con σ_k), con los cuartiles
    definidos en cada sorteo dentro de su posición sobre los mismos jugadores;
  - se evalúa en los elegibles en t + k con el Brier score (frente a la tasa base de 0,25) y una curva de
    confiabilidad de 5 bins.
- **Tabla práctica:** P(cuartil superior en t + k | cuartil de t), observada frente a modelada.

### P4. Atrición y cobertura (REQ-A7, A8)

- **Situación en t + k** de los elegibles en t: elegible, presente pero no elegible, o ausente. Se compara por edad,
  posición, s_t, cobertura del club y minutos.
- **IPW:**
  - logística de elegible_{t+k} sobre s_t, edad, edad², posición, log(minutos_t) y estrato de cobertura del club en t;
  - pesos estabilizados y truncados en el percentil 99;
  - ρ ponderado = correlación ponderada de rangos medios dentro de posición.
  - Se reporta junto al no ponderado para P1 (k = 1 B→C; k = 2 A→C).
- **Estratos de cobertura** en A→B y B→C: club de t con 20 frente a 38 partidos cubiertos.

### P5. Validez externa (REQ-A9; D-13)

- **(a) Minutos futuros:** ρ_W(s_t, proporción de minutos del club en t + 1) entre los elegibles en t presentes en
  t + 1. Además, un Spearman parcial que descuenta edad y edad² dentro de posición. Se calcula en A→B y B→C.
- **(b) Permanencia:** AUC de s_t para predecir la presencia en la liga en t + 1, entre los elegibles en t,
  **excluyendo** a los jugadores cuyo club principal descendió al final de t:
  - A: Leicester City, Leeds United y Southampton;
  - B: Burnley, Luton Town y Sheffield United.
- **(c) Valor de mercado:**
  - Spearman parcial de s_A con log(valor de mercado en B), ajustando por edad y posición;
  - lo mismo con Δ log(valor de mercado) entre A y B.
  - Solo hay valor de mercado de A y B.

### P6. Horizonte implícito frente al observado (REQ-A6; D-12)

- **Para M2:**
  - ρ_W(ŝ_k, s_t) (implícito) frente a ρ_W(s_{t+k}, s_t) (persistencia observada) y ρ_W(ŝ_k, s_{t+k}) (precisión);
  - k = 1 desde B y k = 2 desde A.
- **Reproducción del enviado:** ρk = 0,9402 / 0,7204 / 0,4193 / 0,2045 / 0,0839 desde `outputs/pipeline`, más los
  contrafactuales sin edad y con edad escalada.

### P7. Efectos de edad (REQ-A12)

Coeficientes de M2 con IC 95 % por bootstrap de pares (B = 2.000) y curva marginal del componente de edad con banda.
**No se reportan edades pico.**

### P8. Sensibilidades (REQ-A10, A11)

Sobre P1 y P2 (k = 1 y k = 2):
- pesos iguales;
- posición canónica;
- umbrales de 500 y 750 minutos, y ≥ 30 % de los minutos del club;
- macro 0,5;
- con arqueros;
- solo jugadores con el mismo grupo en t y t + k;
- tramos de exposición: 1.000–1.999 frente a ≥ 2.000 minutos en t;
- indicador en modo legacy con elegibilidad por minutos reales.

### P9. PSR (D-02)

No se re-estima. El apéndice cita el diagnóstico `d06` de la versión enviada y la tabla de `under_pressure` por tipo y
temporada de `data_audit_R1.md`.

## 3. Inferencia

- **Bootstrap:** B = 2.000, remuestreo de jugadores con reemplazo estratificado por posición, semilla 20261108.
- **IC:** percentil 2,5–97,5.
- **Comparaciones pareadas:** con los mismos índices de remuestreo.
- **Sin corrección por multiplicidad:** las métricas principales son tres (H1–H3) y las demás se rotulan como
  secundarias.

## 4. Métricas principales

- **H1.** ρ_W observado a k = 1 (B→C) y k = 2 (A→C).
- **H2.** Δρ_W = ρ_W(M2) − ρ_W(M0) a k = 1 y a k = 2.
- **H3.** Cobertura del intervalo gaussiano de 80 % a k = 1.

## 5. Criterios de decisión en G3 (SPEC §6, HC-3)

Se reportan tal como salgan:

- **R-a.** La cota inferior del IC de ρ por posición a k = 2 es ≤ 0 en DEF, MID y FWD, **y** el IC de Δρ_W a k = 1
  incluye 0 para M2 y M3. Contribución delgada: discutir el formato.
- **R-b.** El IC de ρ_W(s_t, minutos en t + 1) incluye 0 o está por debajo, en ambas transiciones. Hay que
  replantear el constructo.
- **R-c.** El ρ_W principal a k = 1 cambia en más de 0,10 entre el indicador legacy y el de revisión. Hay que
  explicar el origen.
- **R-d.** La cobertura del intervalo a k = 1 se aleja más de 10 puntos del 80 %. Hay que revisar el método.

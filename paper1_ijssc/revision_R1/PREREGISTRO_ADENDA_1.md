# Preregistro: adenda 1 (HC3-2 y HC3-5)

Fecha: 2026-11-08. Se commitea **antes** de construir el indicador de volumen, de enlazar Transfermarkt y de calcular
cualquier métrica de esta adenda. Complementa a `PREREGISTRO.md` (commit `cd5dcdc`) sin modificar nada de él.

## 0. Por qué existe y qué declara

- **Es posterior a un resultado.** El preregistro principal arrojó validez externa nula para el indicador de
  eficiencia (criterio R-b activo; `results/RESULTADOS_R1.md` §6). Los autores decidieron (HC3-2):
  1. reportar ese resultado como límite de constructo;
  2. contrastarlo con una versión de **volumen** del mismo dato.

  El contraste se motiva en el resultado de R-b, y así se declarará en el paper.
- **Datos de valor de mercado.** No hay temporadas nuevas de StatsBomb (HC3-5). La validez externa por valor de
  mercado se extiende con datos de Transfermarkt ya descargados para otro proyecto (Paper A). No se hace scraping.
- **Lo que se vio antes de escribir esta adenda:** solo la existencia y la cobertura de esas tablas (temporadas,
  clubes, filas con valor). Ninguna asociación con el indicador.

## 1. Indicador de volumen (contraste)

- **Definición:** `score_obv_p90`. Es la suma de `obv_total_net` de todos los eventos del jugador en la temporada,
  dividida por sus minutos reales en partidos cubiertos y multiplicada por 90. Se percentiliza (rank promedio,
  0–100) dentro de (temporada, grupo de la temporada), entre los jugadores con ≥ 1.000 minutos.
- **Universo:** el mismo del indicador principal. La elegibilidad (D-04), las posiciones (D-06), la edad (D-05) y los
  arqueros fuera (D-03) no cambian.
- **No tiene parámetros estimados:** ni pesos ni selección de eventos. Solo los minutos y la percentilización.
- **Diferencia con el principal (`score_obv_ar1`):** el principal promedia la OBV **por acción** dentro de cada tipo
  de evento (eficiencia). Este mide la **contribución total por minuto** (volumen × eficiencia).

## 2. Enlace StatsBomb ↔ Transfermarkt y valor de mercado

- **Fuente de planteles:** `whoscored_data/whoscored-scraper/data/whoscored.sqlite`, tabla `tm_squad` (todas las ligas
  descargadas), con `season_id` de Transfermarkt igual al año de inicio de la temporada.
- **Fuente de valoraciones:** `whoscored_data/analisis/datos_externos/tm_valor_historial/tm_valor_historial.parquet`
  (historial por ID de Transfermarkt).
- **Regla de enlace.** Para cada jugador de StatsBomb con fecha de nacimiento, candidatos de Transfermarkt con **la
  misma fecha de nacimiento** en cualquier `tm_squad` de las temporadas 2022–2025.
  - Puntaje de nombre: máximo Jaccard de tokens normalizados (sin tildes, minúsculas) entre {nombre de StatsBomb,
    nombre de uso común} y el nombre de Transfermarkt.
  - Se acepta el candidato de mayor puntaje si es ≥ 0,25. En empate, decide la coincidencia de club en alguna
    temporada.
  - Se reportan la tasa de enlace y una muestra de 30 pares para revisión manual.
  - Los no enlazados quedan fuera de los análisis de valor de mercado.
- **Valor de fin de temporada t (VM_t):** la última valoración de `tm_valor_historial` con fecha ≤ 30 de junio del año
  de término de t. Si el jugador no está en ese historial, se usa `market_value_eur` de su fila de `tm_squad` en la
  temporada t.
- **Temporadas:** A (30/06/2023), B (30/06/2024), C (30/06/2025) y D′ (30/06/2026; solo valor de mercado, sin
  StatsBomb).

## 3. Análisis

Mismas métricas, bootstrap (B = 2.000, estratificado por posición, semilla 20261108) y elegibilidad que el
preregistro principal.

### E1. Indicador de volumen: P1, P2 y P5 del preregistro principal

- **Persistencia:** ρ_W a k = 1 (B→C) y k = 2 (A→C).
- **Precisión:** M0 frente a M2, entrenado en A→B con el indicador de volumen; Δρ_W a k = 1 y k = 2.
- **Validez externa:** minutos en t+1 (ρ_W y parcial por edad), AUC de permanencia (sin descendidos), en A→B y B→C.

### E2. Valor de mercado ampliado, para ambos indicadores

En cada transición A→B, B→C y C→D′, sobre los elegibles en t que tienen VM_t y VM_{t+1}:
- **(i) Nivel:** Spearman parcial del score en t con log VM_{t+1}, ajustando por edad, edad² y dummies de posición.
- **(ii) Cambio:** Spearman parcial del score en t con Δ log VM (t → t+1), con el mismo ajuste.
- **(iii) Nivel contemporáneo:** Spearman parcial del score en t con log VM_t, como referencia de construcción.

### E3. Contrastes pareados volumen − eficiencia (mismos remuestreos)

- Δ en la persistencia ρ_W a k = 1 y k = 2.
- Δ en ρ_W con los minutos en t+1 (A→B y B→C).
- Δ en el Spearman parcial con log VM_{t+1} (B→C y C→D′).
- Además, la correlación entre los dos indicadores dentro de la misma temporada (ρ_W).

## 4. Regla de recomendación (la decisión final es de los autores, HC3-2)

- **Si el volumen muestra validez externa** (IC de ρ_W con los minutos en t+1 o del parcial con log VM_{t+1} por
  encima de 0 en al menos dos de las transiciones evaluadas) **y supera a la eficiencia** (IC del contraste pareado
  por encima de 0 en al menos una): se recomienda reportar ambos indicadores y encuadrar el paper en el contraste
  eficiencia frente a contribución, con la persistencia como eje.
- **Si ninguno muestra validez externa:** se recomienda la opción (a), un paper sobre persistencia y predictibilidad
  de un indicador de eficiencia por acción, con el límite de constructo explícito y el volumen en el suplemento.
- **Si ambos la muestran y no difieren:** se recomienda mantener el indicador principal y reportar el volumen como
  sensibilidad.

En los tres casos, todo lo de esta adenda se rotula en el paper como análisis posterior al resultado R-b.

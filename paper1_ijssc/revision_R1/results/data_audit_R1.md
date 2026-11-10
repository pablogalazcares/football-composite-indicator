# Auditoría de la capa de datos R1 (T1.8)

Generado por `pipeline/r05b_data_audit.py`. Solo descriptivos por temporada; ninguna métrica entre temporadas (se calculan después del preregistro).

## 1. Cobertura

| Temporada | Partidos indexados | Con eventos | Clubes por partidos cubiertos |
|---|---|---|---|
| A (2022/2023) | 290 | 290 | 10 clubes con 20, 10 clubes con 38 |
| B (2023/2024) | 290 | 290 | 10 clubes con 20, 10 clubes con 38 |
| C (2024/2025) | 378 | 378 | 4 clubes con 37, 16 clubes con 38 |

En 2024/25 la entrega trae 378 de los 380 partidos: los 2 faltantes no están en el índice.

## 2. Minutos

- **Reconstrucción validada contra lineups** (A/B): error mediano 0.00 / 0.00 min por jugador-partido; ρ de totales 1.0000 / 0.9999. Detalle en `minutes_validation.md`.
- **Hallazgo sobre la versión enviada.** Los minutos de `data/processed/modeling/player_season_features.parquet`, que usó P04 para el umbral de 1.000, están inflados frente a los reales:
  - razón mediana (enviado / real) 1.23 y p90 3.98;
  - el origen probable es la inferencia `max(minute) + 1` por partido (`src/data_processing/frames_processor.py`), que a un suplente le acredita casi el partido completo;
  - con minutos reales, ≥ 1.000 minutos tienen 270 jugadores en A (no 351) y 271 en B (no 337);
  - de los 211 pares del paper, **solo 160** cumplen ≥ 1.000 minutos reales en ambas temporadas.
  - Afecta la muestra de entrenamiento (N = 211) y todo lo que depende de ella. Debe declararse en la respuesta a M7.

## 3. Jugadores

| Temporada | Jugadores con minutos | Elegibles (≥ 1.000) | Sin edad (todos / elegibles) | Origen de la fecha de nacimiento | Con más de un club |
|---|---|---|---|---|---|
| A | 550 | 270 | 0 / 0 | {'lineup': 550} | 14 |
| B | 560 | 274 | 0 / 0 | {'lineup': 560} | 10 |
| C | 560 | 321 | 1 / 0 | {'lineup': 395, 'players_master': 164, 'missing': 1} | 12 |

Cambios de grupo posicional (modal por temporada):
- A->B: 37 de 346 jugadores presentes en ambas; 7 de 161 elegibles en ambas.
- B->C: 24 de 349 jugadores presentes en ambas; 8 de 183 elegibles en ambas.
- En A/B, el grupo de la temporada difiere del canónico de P01 en 65 jugador-temporadas.

Casos señalados por los revisores o por el diagnóstico (capa nueva):

| Temporada | Jugador | Minutos | Partidos | Club | Elegible | Score |
|---|---|---|---|---|---|---|
| A | David Raya | 3760 | 38 | Brentford | sí | 65.4 |
| B | David Raya | 3234 | 32 | Arsenal | sí | 68.1 |
| C | David Raya | 3686 | 37 | Arsenal | sí | 71.0 |
| A | Ethan Nwaneri | 4 | 1 | Arsenal | no | 32.0 |
| B | Ethan Nwaneri | 17 | 1 | Arsenal | no | 44.5 |
| C | Ethan Nwaneri | 1018 | 25 | Arsenal | sí | 48.0 |
| A | Jonny Castro | 564 | 8 | Wolverhampton Wanderers | no | 43.9 |
| B | Jonny Castro | 11 | 1 | Wolverhampton Wanderers | no | 94.4 |
| A | Oriol Romeu | 77 | 1 | Southampton | no | 78.3 |
| B | Ryan Gravenberch | 1275 | 26 | Liverpool | sí | 48.4 |
| C | Ryan Gravenberch | 3355 | 36 | Liverpool | sí | 48.6 |
| A | Shea Charles | 33 | 1 | Manchester City | no | 43.3 |

Jugadores con más de un club en la temporada: 36. El club principal es el de más minutos; la lista completa está en `players_season.parquet` (columna `clubs`).

## 4. Comparabilidad entre temporadas

Proporción de eventos con `under_pressure` por tipo:

| Tipo | A | B | C |
|---|---|---|---|
| Carry | 0.322 | 0.334 | 0.367 |
| Clearance | 1.000 | 1.000 | 1.000 |
| Dribble | 1.000 | 1.000 | 1.000 |
| Duel | 1.000 | 1.000 | 1.000 |
| Pass | 0.155 | 0.163 | 0.183 |
| Shot | 0.355 | 0.299 | 0.357 |

Duel, Dribble y Clearance llevan `under_pressure` en el 100 % de los casos, por definición de StatsBomb (Data Specification v1.1, Appendix 7).

OBV FOR por tipo (media, DE, decimales de la entrega):

| Temporada | Tipo | Media | DE | Decimales (mediana) |
|---|---|---|---|---|
| A | Pass | 0.00157 | 0.01872 | 10 |
| A | Carry | 0.0018 | 0.01512 | 10 |
| A | Shot | -0.00504 | 0.16513 | 10 |
| B | Pass | 0.00158 | 0.01885 | 10 |
| B | Carry | 0.002 | 0.01578 | 10 |
| B | Shot | -0.00192 | 0.17053 | 10 |
| C | Pass | 0.00145 | 0.01846 | 4 |
| C | Carry | 0.00202 | 0.01529 | 4 |
| C | Shot | -0.00321 | 0.16376 | 4 |

La entrega 2024/25 trae OBV redondeado a 4 decimales. Como el indicador usa percentiles dentro de la temporada, el redondeo y cualquier cambio de nivel del modelo OBV entre entregas afectan solo los empates; igual se declara.

## 5. Indicador

- **Modo legacy:** reproduce `score_OBV` de P03 en 1100 jugador-temporadas, con diferencia máxima 2.8e-14 y el mismo patrón de faltantes. Las medias por tipo desde el crudo coinciden con p02 (8733 filas, diferencia 0.0).
- **Pesos AR1** re-estimados desde crudo solo con A→B: reproducen las constantes de P03 (diferencia máxima 5.0e-04; mismos conjuntos de eventos).
- **Modo revisión frente a legacy en A/B** (mismo jugador-temporada): Spearman 0.980, diferencia máxima 37.0 puntos. La causa es la posición de la temporada en vez de la canónica (cambia el grupo del percentil).
- Jugador-temporadas con score en modo revisión: {'A': 546, 'B': 557, 'C': 560}.

## 6. Muestras (D-04: ≥ 1.000 minutos reales)

| Transición | Con score en t | Elegibles en t | Elegibles en ambas | Elegibles en t ausentes en t+k | Presentes pero no elegibles en t+k |
|---|---|---|---|---|---|
| A->B | 546 | 270 | 161 | 51 | 58 |
| A->C | 546 | 270 | 147 | 67 | 56 |
| B->C | 557 | 274 | 183 | 39 | 52 |

Elegibles en las tres temporadas: 117. Universo del panel (base 2024/25, elegibles): 321.

Elegibles por temporada y posición: {"A": {"DEF": 108, "FWD": 58, "GK": 23, "MID": 81}, "B": {"DEF": 107, "FWD": 67, "GK": 24, "MID": 76}, "C": {"DEF": 123, "FWD": 79, "GK": 23, "MID": 96}}

## 7. Criterios de bloqueo de HC-1

- REQ-D2 (minutos): cumple
- REQ-D4 (≤ 1 % de elegibles sin edad): cumple
- REQ-C1 (legacy = P03): cumple

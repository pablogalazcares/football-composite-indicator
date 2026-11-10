# Validación de minutos reconstruidos desde eventos (REQ-D2)

Método: titulares por Starting XI, entradas y salidas por Substitution, salidas por roja o segunda amarilla, fin de período por Half End; tiempo transcurrido por período, incluido el descuento. Referencia: `positions` de los lineups de StatsBomb (solo existen en 2022/23 y 2023/24).

Criterios: error absoluto mediano ≤ 2.0 min por jugador-partido; Spearman de totales por temporada ≥ 0.99.

| Temporada | Jugador-partidos (eventos / lineups) | Solo en eventos | Solo en lineups | Error mediano (min) | Error p95 (min) | % > 10 min | ρ totales | Δ media total (min) | Cumple |
|---|---|---|---|---|---|---|---|---|---|
| A | 8653 / 8653 | 0 | 0 | 0.00 | 0.00 | 0.16 | 1.0000 | -0.7 | sí |
| B | 8689 / 8689 | 0 | 0 | 0.00 | 0.00 | 0.41 | 0.9999 | -2.2 | sí |

La temporada 2024/25 no tiene lineups: sus minutos salen del mismo método validado aquí.

Discrepancias > 10 min: `outputs/r1/minutes_discrepancies_<temporada>.csv`.

## Exposición por cobertura del club

| Temporada | Partidos cubiertos del club | Jugador-temporadas | ≥ 1.000 min | % ≥ 1.000 |
|---|---|---|---|---|
| A | 20 | 274 | 105 | 38.3 |
| A | 38 | 276 | 165 | 59.8 |
| B | 20 | 278 | 104 | 37.4 |
| B | 38 | 282 | 170 | 60.3 |
| C | 37 | 110 | 66 | 60.0 |
| C | 38 | 450 | 255 | 56.7 |

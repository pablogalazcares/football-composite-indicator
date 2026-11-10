# Preregistro: adenda 2 (robustez pedida por la lectura adversaria simulada, T3.8)

Fecha: 2026-11-09. Se commitea **antes** de calcular cualquiera de los análisis de abajo. Complementa a
`PREREGISTRO.md` (`cd5dcdc`) y a `PREREGISTRO_ADENDA_1.md` (`c431da2`) sin modificarlos.

## 0. Por qué existe y qué declara

- **Es posterior a todos los resultados** del preregistro y de la adenda 1, que los autores ya conocen.
- Responde a dos lecturas adversarias simuladas de la versión revisada (`results/simulated_review_R1.md`). Ahí se
  señala que:
  - el contraste eficiencia–contribución es post hoc;
  - la contribución por 90 minutos puede reflejar el contexto del club (posesión, calidad del equipo);
  - la asociación con el valor de mercado no está ajustada por club;
  - la Tabla 3 usa una definición de cuartiles distinta de la preregistrada;
  - los intervalos de Δρ son condicionales al modelo ajustado;
  - el modelo con edad se ajusta y evalúa solo sobre los que siguen elegibles.
- **Todo lo de esta adenda es exploratorio o de robustez.** Se reporta en el suplemento cualquiera sea el resultado y
  no reemplaza ninguna cifra principal.
- **Lo que se vio antes de escribirla:** todas las cifras de `results/numbers.tex` (600 macros). Ninguna de las
  cantidades R1–R7 de abajo.

Mismos datos, elegibilidad, posiciones, bootstrap (B = 2.000, estratificado por posición, semilla 20261108) y métrica
ρ_W que el preregistro principal. Los dos resúmenes son `score_obv_ar1` (eficiencia) y `score_obv_p90` (contribución).

## 1. Análisis

### R1. Contexto del club en la persistencia
- **(a) Quienes se quedan y quienes cambian.** Se separan según si el club principal en t es igual al de t+k. Para
  cada resumen, se calcula ρ_W con IC a k = 1 (B→C) y k = 2 (A→C), con el n de cada grupo. Además, la diferencia
  quedarse − cambiar, por bootstrap no pareado.
- **(b) Persistencia dentro de club.** Spearman parcial entre el score en t y en t+k, ajustando por dummies de
  posición, del club principal en t y del club principal en t+k (rangos residualizados por MCO). Va para ambos
  resúmenes, con la diferencia pareada contribución − eficiencia y su IC.

### R2. Valor de mercado ajustado por club
Se repite E2 de la adenda 1 (nivel siguiente, nivel contemporáneo y cambio; transiciones A→B, B→C y C→D′) agregando
dummies del club principal en t al ajuste (edad, edad², posición). Va para ambos resúmenes, con la diferencia pareada
contribución − eficiencia.

### R3. Retención del cuartil superior con la definición preregistrada
- Cuartiles en t dentro del grupo de posición, sobre los mismos jugadores evaluados (elegibles en t y t+k: n = 168 y
  134).
- Para cada cuartil en t: frecuencia observada de estar en el cuartil superior en t+k, con n e IC por bootstrap.
  Va para ambos resúmenes.
- Para la eficiencia se agrega la probabilidad media del modelo M2, simulada como en `r08`, para los mismos jugadores.

### R4. Δρ ponderado por la probabilidad inversa de seguir elegible
ρ_W(M2) − ρ_W(M0) a k = 1 y k = 2 con los pesos IPW de `r09`, con correlación de rangos ponderada e IC por bootstrap.
Va para la eficiencia y la contribución.

### R5. Bootstrap conjunto de entrenamiento y evaluación
En cada réplica se remuestrean los pares de entrenamiento A→B (estratificados por posición), se reajusta M2 por MCO, y
se remuestrean los jugadores de evaluación. Se calcula Δρ_W = ρ_W(M2) − ρ_W(M0) a k = 1 y k = 2, con IC percentil.
Va para la eficiencia y la contribución.

### R6. Escala de los resúmenes
- Desviación estándar, mínimo y máximo de cada resumen entre los elegibles de cada temporada.
- El MAE y el semiancho del intervalo del 80 %, expresados en desviaciones estándar de la temporada evaluada.

### R7. Calibración
Tabla de confiabilidad de P(cuartil superior) de la eficiencia, a k = 1 y k = 2, con todos los bins (incluidos los de
n < 5), su n e IC de Wilson de la frecuencia observada.

## 2. Cómo se interpreta (fijado antes de ver resultados)

- **R1 y R2.** El contraste eficiencia–contribución se describirá como "robusto al club" solo si **ambas** condiciones
  se cumplen:
  - en R1(b), la diferencia pareada a k = 2 conserva al menos la mitad de su valor sin ajustar (0,21) y su IC no cruza
    cero;
  - en R2, el parcial de la contribución con log VM_{t+1} conserva IC sobre cero en al menos dos de las tres
    transiciones.

  Si no se cumplen, el paper dirá que el contraste puede reflejar en parte el contexto del club.
- **R3 a R7** reemplazan o complementan la presentación de cifras ya reportadas; no cambian conclusiones por sí
  solos. Si R5 da intervalos que incluyen valores positivos relevantes (límite superior > 0,05), el texto dirá que no
  se puede descartar una mejora pequeña del modelo con edad.

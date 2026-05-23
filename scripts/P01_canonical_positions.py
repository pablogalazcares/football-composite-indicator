"""
P01 — Canonical Player Positions
=================================
Objetivo:
    Construir la única fuente de verdad de posición para todos los jugadores EPL.
    Produce un archivo con dos niveles de posición:
      - position_detail: posición granular StatsBomb (ej. "Right Center Back")
      - position_group:  macro-grupo MIP (GK / DEF / MID / FWD)

    Elimina completamente la dependencia del join OBV que producía 46% de inflación
    en MID en el pipeline anterior (bug identificado en sesión 2026-05-16).

Fuente primaria:
    data/processed/statsbomb_legacy/df_positions.csv
    Contiene la posición modal de cada jugador a partir de TODAS sus apariciones
    en EPL 2022/23 + 2023/24. Derivado de StatsBomb lineups, es la fuente más
    representativa disponible. Tiene player_id, player_name, position, max_perc.

Fallback para no-coincidencias:
    Si un jugador del universo no aparece en df_positions.csv (por player_id),
    se intenta resolver via:
      1. Matching por nombre normalizado (Unicode + minúsculas)
      2. Fuzzy matching Jaccard (threshold >= 0.60) sobre tokens del nombre
      3. [Opcional / activable] Búsqueda web via Transfermarkt:
         https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche?query=<nombre>
         Para activar: set USE_TM_WEB_FALLBACK = True en config.

Decisiones tomadas:
    - df_positions.csv es la fuente primaria (no la columna "position" en PSF,
      que viene de un join OBV y tiene 175 nulls en EPL 2023/24).
    - Se guardan AMBOS niveles: position_detail y position_group.
      Razón: future work incluye modelos por posición específica (p.ej. separar
      RCB de LCB, o Center Forward de Second Striker).
    - Mapeo granular → macro basado en criterio morfológico-táctico estándar.
    - Marcus Bettinelli (player_id=12258, portero): único caso no cubierto por
      df_positions.csv — imputado como GK explícitamente.

DECISIONES PENDIENTES:
    - USE_TM_WEB_FALLBACK: actualmente False. Activar si en iteraciones futuras
      hay nuevas incorporaciones de jugadores no presentes en el legacy EPL.
    - ¿Usar la posición de 2023/24 solamente, o la modal combinada 22/23+23/24?
      Actualmente se usa la modal combinada (df_positions.csv). Para tracking
      de cambios de posición en el tiempo, se necesitaría una versión por temporada.

Outputs:
    outputs/pipeline/p01_canonical_positions.parquet
    outputs/pipeline/p01_position_coverage.csv  — reporte de cobertura y fallbacks usados

Tests inline:
    - Distribución de macro-grupos (MID < 35% — test de regresión del bug)
    - 0 nulos en position_group
    - Sin player_id duplicados
    - Todos los valores de POS_MAP están cubiertos

Uso downstream:
    P03 (cálculo de percentiles por position_group)
    P06 (composición del universo de proyección)
    P09 (imputation salarial por position_group)
    P10 (input del optimizador)
"""

from __future__ import annotations

import unicodedata
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

# ── Configuración ──────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[2]
POS_CSV  = REPO / "data/processed/statsbomb_legacy/df_positions.csv"
MV_CSV   = REPO / "data/processed/statsbomb_legacy/df_final_market_value.csv"
OUT_DIR  = REPO / "outputs/pipeline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

USE_TM_WEB_FALLBACK = False  # DECISIÓN PENDIENTE: activar si se agregan jugadores nuevos

# Mapeo posición StatsBomb granular → macro-grupo (GK / DEF / MID / FWD)
# Clave de diseño: todos los valores posibles en df_events_obv["position"] deben estar aquí.
POS_MAP: dict[str, str] = {
    # Porteros
    "Goalkeeper": "GK",
    # Defensas
    "Right Back": "DEF", "Left Back": "DEF",
    "Right Center Back": "DEF", "Left Center Back": "DEF",
    "Center Back": "DEF", "Sweeper": "DEF",
    "Right Wing Back": "DEF", "Left Wing Back": "DEF",
    # Mediocampistas
    "Defensive Midfield": "MID", "Center Midfield": "MID",
    "Left Center Midfield": "MID", "Right Center Midfield": "MID",
    "Attacking Midfield": "MID",
    "Left Midfield": "MID", "Right Midfield": "MID",
    "Left Defensive Midfield": "MID", "Right Defensive Midfield": "MID",
    "Center Defensive Midfield": "MID", "Center Attacking Midfield": "MID",
    # Delanteros
    "Right Wing": "FWD", "Left Wing": "FWD",
    "Center Forward": "FWD", "Second Striker": "FWD",
    "Left Center Forward": "FWD", "Right Center Forward": "FWD",
}

# Casos de imputación explícita (no resolubles via df_positions.csv por ausencia)
EXPLICIT_IMPUTATIONS: dict[int, tuple[str, str]] = {
    12258: ("Goalkeeper", "GK"),  # Marcus Bettinelli — único jugador fuera de df_positions
}


def _normalize_name(name: str) -> str:
    """Normaliza a minúsculas, sin diacríticos, para matching de nombres."""
    nfkd = unicodedata.normalize("NFKD", str(name))
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def _jaccard_tokens(a: str, b: str) -> float:
    """Similitud de Jaccard sobre tokens de dos strings normalizados."""
    ta, tb = set(_normalize_name(a).split()), set(_normalize_name(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def load_positions() -> pd.DataFrame:
    """Carga df_positions.csv y valida estructura mínima."""
    assert POS_CSV.exists(), f"df_positions.csv no encontrado en {POS_CSV}"
    df = pd.read_csv(POS_CSV)
    required = {"player_id", "player_name", "position", "max_perc"}
    missing = required - set(df.columns)
    assert not missing, f"Columnas faltantes en df_positions.csv: {missing}"
    assert df["player_id"].isna().sum() == 0, "player_id nulo en df_positions.csv"
    assert df["player_id"].duplicated().sum() == 0, "player_id duplicado en df_positions.csv"
    return df


def validate_pos_map(positions: pd.DataFrame) -> None:
    """Verifica que todos los valores de 'position' están en POS_MAP."""
    unmapped = set(positions["position"].unique()) - set(POS_MAP.keys())
    if unmapped:
        # Warning en lugar de assert: puede haber posiciones raras no vistas antes
        print(f"  [AVISO] Posiciones sin mapeo en POS_MAP: {unmapped}")
        print(f"  Estas filas recibirán position_group=None y deberán revisarse.")


def build_canonical(positions: pd.DataFrame) -> pd.DataFrame:
    """Construye el DataFrame canónico con ambos niveles de posición."""
    out = positions[["player_id", "player_name", "position", "max_perc"]].copy()
    out = out.rename(columns={"position": "position_detail"})
    out["position_group"] = out["position_detail"].map(POS_MAP)
    return out


def apply_explicit_imputations(out: pd.DataFrame) -> pd.DataFrame:
    """Aplica imputaciones explícitas para casos conocidos sin cobertura."""
    for pid, (detail, group) in EXPLICIT_IMPUTATIONS.items():
        if pid not in out["player_id"].values:
            new_row = {
                "player_id": pid,
                "player_name": f"[imputed pid={pid}]",
                "position_detail": detail,
                "position_group": group,
                "max_perc": 1.0,
                "assignment_method": "explicit_imputation",
            }
            out = pd.concat([out, pd.DataFrame([new_row])], ignore_index=True)
        else:
            mask = out["player_id"] == pid
            out.loc[mask, "position_detail"] = detail
            out.loc[mask, "position_group"] = group
    return out


def resolve_missing(out: pd.DataFrame, universe_pids: list[int]) -> pd.DataFrame:
    """
    Para player_ids del universo que no están en df_positions.csv,
    intenta resolver por nombre (normalización + Jaccard) y opcionalmente TM web.
    """
    covered = set(out["player_id"].tolist())
    missing_pids = [p for p in universe_pids if p not in covered]
    if not missing_pids:
        return out

    print(f"  [INFO] {len(missing_pids)} player_ids del universo sin posición canónica")
    print(f"  Intentando fallback por nombre...")

    # Cargar df_final_market_value para obtener nombres de los faltantes
    mv = pd.read_csv(MV_CSV)
    mv_idx = mv.drop_duplicates("player_id").set_index("player_id")

    pos_names_norm = {
        _normalize_name(row["player_name"]): row
        for _, row in out.iterrows()
    }

    rows_resolved = []
    rows_unresolved = []

    for pid in missing_pids:
        if pid not in mv_idx.index:
            rows_unresolved.append(pid)
            continue

        name = str(mv_idx.loc[pid, "player_name"])
        name_norm = _normalize_name(name)

        # Intento 1: match exacto normalizado
        if name_norm in pos_names_norm:
            matched = pos_names_norm[name_norm]
            rows_resolved.append({
                "player_id": pid, "player_name": name,
                "position_detail": matched["position_detail"],
                "position_group": matched["position_group"],
                "max_perc": matched.get("max_perc", 1.0),
                "assignment_method": "name_exact_fallback",
            })
            continue

        # Intento 2: Jaccard sobre tokens
        best_score, best_row = 0.0, None
        for norm_name, row in pos_names_norm.items():
            j = _jaccard_tokens(name, norm_name)
            if j > best_score:
                best_score, best_row = j, row

        if best_score >= 0.60 and best_row is not None:
            rows_resolved.append({
                "player_id": pid, "player_name": name,
                "position_detail": best_row["position_detail"],
                "position_group": best_row["position_group"],
                "max_perc": best_row.get("max_perc", 1.0),
                "assignment_method": f"jaccard_{best_score:.2f}_fallback",
            })
        elif USE_TM_WEB_FALLBACK:
            # DECISIÓN PENDIENTE: implementar búsqueda web TM si se habilita
            # URL: https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche?query=<name>
            # Parsear primer resultado → link de jugador → extraer posición
            print(f"  [TM_WEB_FALLBACK] pid={pid} {name} — no implementado aún")
            rows_unresolved.append(pid)
        else:
            rows_unresolved.append(pid)

    if rows_resolved:
        out = pd.concat([out, pd.DataFrame(rows_resolved)], ignore_index=True)

    if rows_unresolved:
        print(f"  [AVISO] {len(rows_unresolved)} jugadores sin posición resuelta: {rows_unresolved}")
        print(f"  Estos no tendrán posición asignada. Considerar activar USE_TM_WEB_FALLBACK.")

    return out


def validate_output(out: pd.DataFrame) -> None:
    """Tests de salida — fallan el script si hay regresiones."""
    assert out["player_id"].duplicated().sum() == 0, \
        f"player_id duplicado en output canonical_positions"
    assert out["position_group"].isna().sum() == 0, \
        f"position_group nulo: {out[out['position_group'].isna()][['player_id','player_name']].to_string()}"
    assert set(out["position_group"].unique()).issubset({"GK", "DEF", "MID", "FWD"}), \
        f"Valores inesperados en position_group: {out['position_group'].unique()}"

    mid_pct = (out["position_group"] == "MID").mean()
    assert mid_pct < 0.35, (
        f"TEST DE REGRESIÓN FALLIDO: MID={mid_pct:.1%} — supera el 35%.\n"
        f"  Causa probable: imputación por defecto a MID (bug del pipeline anterior).\n"
        f"  Distribucion esperada según df_positions.csv: GK~7%, DEF~34%, MID~27%, FWD~32%."
    )

    gk_pct = (out["position_group"] == "GK").mean()
    assert 0.04 < gk_pct < 0.12, \
        f"GK={gk_pct:.1%} fuera del rango esperado 4–12%"

    print(f"\n  [TESTS OK] Distribución position_group:")
    for pos in ["GK", "DEF", "MID", "FWD"]:
        n = (out["position_group"] == pos).sum()
        print(f"    {pos}: {n} ({n/len(out):.1%})")


def generate_coverage_report(out: pd.DataFrame, total_in_source: int) -> pd.DataFrame:
    """Genera reporte de cobertura y métodos de asignación."""
    method_counts = out["assignment_method"].value_counts().reset_index()
    method_counts.columns = ["method", "n_players"]
    method_counts["pct"] = method_counts["n_players"] / len(out) * 100
    return method_counts


def main() -> None:
    print("=" * 60)
    print("P01 — Canonical Player Positions")
    print("=" * 60)

    # 1. Cargar y validar input
    positions = load_positions()
    total_source = len(positions)
    print(f"\n  df_positions.csv: {total_source} jugadores EPL")

    # 2. Verificar mapeo
    validate_pos_map(positions)

    # 3. Construir canónico
    out = build_canonical(positions)
    out["assignment_method"] = "df_positions_primary"

    # 4. Imputaciones explícitas
    out = apply_explicit_imputations(out)

    # 5. Resolver faltantes respecto al universo de proyección
    # (al momento de P01 no tenemos el universo final de P06, usamos MV como proxy)
    mv = pd.read_csv(MV_CSV)
    universe_pids = mv["player_id"].dropna().astype(int).unique().tolist()
    out = resolve_missing(out, universe_pids)

    # 6. Validar output
    validate_output(out)

    # 7. Guardar
    out_path = OUT_DIR / "p01_canonical_positions.parquet"
    out.to_parquet(out_path, index=False)

    coverage = generate_coverage_report(out, total_source)
    cov_path = OUT_DIR / "p01_position_coverage.csv"
    coverage.to_csv(cov_path, index=False)

    print(f"\n  Cobertura:")
    print(coverage.to_string(index=False))
    print(f"\n  Outputs:")
    print(f"    {out_path}")
    print(f"    {cov_path}")
    print(f"\n  Total jugadores con posición canónica: {len(out)}")


if __name__ == "__main__":
    main()

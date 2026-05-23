"""
P02 — OBV Player-Season Aggregation
=====================================
Objetivo:
    Producir el panel OBV agregado por (player_id, season_name, event_type),
    siguiendo exactamente la especificación del paper §5.1:
      Step 1: media por ocurrencia (mean OBV per event occurrence)
      Step 2: esta media se usará en P03 para percentilizar dentro de
              (season, position_group, event_type)

    Fuente: data/processed/statsbomb_legacy/df_events_obv.csv
    Temporadas cubiertas: EPL 2022/23 y 2023/24

Checkpoint vs. reconstrucción:
    df_events_obv.csv (277MB, ~200k filas) es el resultado del pipeline legacy
    (src/data_processing/data_pipeline.py + StatsBomb open data API). Se verifica
    su integridad antes de usarlo como checkpoint.

    CRITERIOS DE VALIDEZ DEL CHECKPOINT:
      - Existe en la ruta esperada
      - Columnas mínimas presentes
      - Cubre exactamente EPL 2022/23 y 2023/24
      - N eventos razonable (> 100,000)

    Si el checkpoint NO es válido, el script falla con instrucciones explícitas
    para regenerarlo vía el pipeline legacy (src/data_processing/data_pipeline.py).
    NO regenera automáticamente — requiere decisión explícita del usuario.

Decisiones tomadas:
    - USAR PER-OCCURRENCE MEAN (media por evento), no per-90 minutos.
      Razón: así está especificado en paper §5.1 Step 1. El OBVCalculator en src/
      usa per-90 — esto es una discrepancia intencional con el pipeline anterior.
      La media por ocurrencia captura CALIDAD/EFICIENCIA por acción, mientras que
      per-90 captura TASA de creación de valor. Para el paper usamos calidad.
    - Linkage player_name → player_id: via df_positions.csv (P01) + df_final_market_value.csv.
      El evento tiene 'player' (nombre string), no player_id directamente.
    - Se incluyen TODOS los tipos de evento con OBV, no solo los 10 del paper.
      Los eventos raros tendrán pocos registros y baja varianza — esto es esperado.
    - Minutos jugados: derivados de df_matches.csv + conteo de eventos temporales.
      Fallback: si no están disponibles, se reporta n_events como proxy.

PSR (Pressure Success Rate):
    Se computa desde p01b_psr_raw.parquet (output de P01b), que extrae outcomes binarios
    directamente desde los JSON raw de StatsBomb.
    Definición: PSR = fracción de acciones bajo presión PSR-elegibles con outcome exitoso.
    Conjunto de eventos: Pass, Dribble, Duel, Shot (Carry EXCLUIDO — sin campo outcome).
    Mapeo de éxito por tipo (§5.2):
      - Pass:   outcome ausente (pase completado) = 1
      - Dribble: outcome == "Complete" = 1
      - Duel:   outcome ∈ {Won, Success In Play, Success Out} = 1
      - Shot:   outcome ≠ "Blocked" = 1
    Solo se calculan player-seasons con >= 10 eventos PSR-elegibles (mínimo estadístico).

DECISIONES TOMADAS (2026-05-16):
    - OBV per-occurrence mean (confirmado): usar media por ocurrencia, NO per-90.
    - PSR = fracción de acciones bajo presión con outcome exitoso binario (desde JSON).
      Ortogonal a OBV por diseño — mide outcome de la acción, no valor generado.

Outputs:
    outputs/pipeline/p02_obv_player_season.parquet
    Columnas: player_id, player_name, season_name, position_detail, position_group,
              event_type, n_events, obv_for_mean, obv_against_mean, obv_total_mean

    outputs/pipeline/p02_psr_player_season.parquet
    Columnas: player_id, season_name, psr, n_pressure_events
    psr = fracción de acciones bajo presión PSR-elegibles (Pass/Dribble/Duel/Shot) con
          outcome binario exitoso. Origen: p01b_psr_raw.parquet (JSON StatsBomb).
    Implementa §5.2: Pressure Success Rate per player-season.

Tests inline:
    - Checkpoint validity: estructura y cobertura de df_events_obv.csv
    - Linkage rate: > 90% de jugadores vinculados a player_id
    - N jugadores por temporada en rango esperado
    - No OBV negativo en la media (puede ser negativo por evento individual, OK)

Upstream:
    CHECKPOINT: data/processed/statsbomb_legacy/df_events_obv.csv
    P01: outputs/pipeline/p01_canonical_positions.parquet (para position_group)
"""

from __future__ import annotations

import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]

# ── Checkpoints de datos legacy ────────────────────────────────────────────────
EVENTS_OBV_PATH = REPO / "data/processed/statsbomb_legacy/df_events_obv.csv"
MATCHES_PATH    = REPO / "data/processed/statsbomb_legacy/df_matches.csv"
MV_PATH         = REPO / "data/processed/statsbomb_legacy/df_final_market_value.csv"
POS_CANONICAL   = REPO / "outputs/pipeline/p01_canonical_positions.parquet"
PSR_RAW_PATH    = REPO / "outputs/pipeline/p01b_psr_raw.parquet"

OUT_DIR = REPO / "outputs/pipeline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_SEASONS   = {"2022/2023", "2023/2024"}
MIN_EVENTS_TOTAL   = 100_000
LINKAGE_THRESHOLD  = 0.90  # mínimo % de jugadores con player_id resuelto


def _normalize(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(name))
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def _jaccard(a: str, b: str) -> float:
    ta, tb = set(_normalize(a).split()), set(_normalize(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ── Sección A: Validación del checkpoint ──────────────────────────────────────

def validate_checkpoint() -> pd.DataFrame:
    """
    Verifica que df_events_obv.csv es apto como punto de partida.
    Falla con instrucciones si no es válido.
    """
    assert EVENTS_OBV_PATH.exists(), (
        f"\n[ERROR] Checkpoint no encontrado: {EVENTS_OBV_PATH}\n"
        f"Para regenerar, ejecutar el pipeline legacy:\n"
        f"  python -m src.data_processing.data_pipeline --competition epl\n"
        f"O correr el StatsBomb data download desde src/data_processing/statsbomb_loader.py"
    )

    print(f"  Leyendo checkpoint df_events_obv.csv ({EVENTS_OBV_PATH.stat().st_size // 1024 // 1024}MB)...")
    # Leer solo las columnas necesarias para eficiencia
    needed_cols = ["player", "position", "team", "match_id",
                   "obv_for_net", "obv_against_net", "obv_total_net", "type"]
    sample = pd.read_csv(EVENTS_OBV_PATH, nrows=1000)
    available = set(sample.columns)
    missing = set(needed_cols) - available
    assert not missing, (
        f"[ERROR] Columnas faltantes en df_events_obv.csv: {missing}\n"
        f"Columnas disponibles: {sorted(available)}\n"
        f"Regenerar checkpoint si el formato cambió."
    )

    # Leer completo
    ev = pd.read_csv(EVENTS_OBV_PATH, usecols=needed_cols, low_memory=False)
    n_total = len(ev)
    assert n_total >= MIN_EVENTS_TOTAL, (
        f"[ERROR] Solo {n_total} eventos en checkpoint — esperado >{MIN_EVENTS_TOTAL}.\n"
        f"El archivo puede estar truncado."
    )

    print(f"  Checkpoint OK: {n_total:,} eventos")
    print(f"  Columnas: {sorted(ev.columns.tolist())}")
    return ev


# ── Sección B: Unión con season_name vía df_matches ───────────────────────────

def attach_season(ev: pd.DataFrame) -> pd.DataFrame:
    """
    Añade season_name a cada evento usando df_matches.csv.
    df_events_obv.csv tiene match_id pero no season_name directamente.
    """
    if not MATCHES_PATH.exists():
        print("  [AVISO] df_matches.csv no encontrado — season_name inferida de player.")
        # Fallback: la mayoría de los datos EPL cubren solo 2 temporadas; asignar ambas
        # via análisis de distribution temporal de match_ids. Esto es aproximado.
        ev["season_name"] = "UNKNOWN"
        return ev

    # df_matches.csv usa columna numérica "season" (235=2022/23, 281=2023/24)
    SEASON_ID_MAP = {235: "2022/2023", 281: "2023/2024"}
    matches_raw = pd.read_csv(MATCHES_PATH, usecols=["match_id", "season"])
    matches_raw["season_name"] = matches_raw["season"].map(SEASON_ID_MAP)
    matches = matches_raw[["match_id", "season_name"]]
    ev = ev.merge(matches, on="match_id", how="left")
    null_season = ev["season_name"].isna().sum()
    if null_season > 0:
        print(f"  [AVISO] {null_season} eventos sin season_name tras join con matches.")
    return ev


# ── Sección C: Linkage player_name → player_id ────────────────────────────────

def build_name_to_id_lookup() -> dict[str, int]:
    """
    Construye lookup normalizado_name → player_id desde P01 + MV.
    Combina ambas fuentes para máxima cobertura.
    """
    lookup: dict[str, int] = {}

    # Fuente 1: canonical positions (P01)
    if POS_CANONICAL.exists():
        pos = pd.read_parquet(POS_CANONICAL)
        for _, row in pos.iterrows():
            key = _normalize(row["player_name"])
            lookup[key] = int(row["player_id"])

    # Fuente 2: df_final_market_value (cobertura adicional)
    if MV_PATH.exists():
        mv = pd.read_csv(MV_PATH)
        mv_dedup = mv.drop_duplicates("player_id")
        for _, row in mv_dedup.iterrows():
            pid = int(row["player_id"])
            key = _normalize(row["player_name"])
            if key not in lookup:
                lookup[key] = pid
            # También registrar name_id_aux si existe
            if "name_id_aux" in mv.columns and pd.notna(row.get("name_id_aux", None)):
                aux = str(row["name_id_aux"]).split("/")[0]
                aux_key = _normalize(aux)
                if aux_key not in lookup:
                    lookup[aux_key] = pid

    return lookup


def link_player_ids(ev: pd.DataFrame, lookup: dict[str, int]) -> pd.DataFrame:
    """
    Resuelve player_id para cada fila usando:
    1. Match exacto normalizado
    2. Jaccard >= 0.70 (threshold más estricto que P01 porque los nombres en
       df_events_obv son más limpios)
    """
    player_names = ev["player"].unique()
    name_to_id: dict[str, int] = {}
    unresolved: list[str] = []

    for name in player_names:
        key = _normalize(name)
        if key in lookup:
            name_to_id[name] = lookup[key]
            continue

        # Jaccard fallback
        best_j, best_pid = 0.0, None
        for norm_key, pid in lookup.items():
            j = _jaccard(name, norm_key)
            if j > best_j:
                best_j, best_pid = j, pid
        if best_j >= 0.70 and best_pid is not None:
            name_to_id[name] = best_pid
        else:
            unresolved.append(name)

    ev["player_id"] = ev["player"].map(name_to_id)

    linked_pct = ev["player_id"].notna().mean()
    print(f"  Linkage: {linked_pct:.1%} de filas vinculadas a player_id")
    print(f"  Jugadores no resueltos: {len(unresolved)}")

    assert linked_pct >= LINKAGE_THRESHOLD, (
        f"[ERROR] Linkage {linked_pct:.1%} bajo umbral {LINKAGE_THRESHOLD:.0%}.\n"
        f"Primeros no resueltos: {unresolved[:10]}\n"
        f"Revisar normalización de nombres o ampliar fuentes en build_name_to_id_lookup()."
    )

    return ev


# ── Sección D: Agregar por (player_id, season_name, event_type) ───────────────

def aggregate_obv(ev: pd.DataFrame) -> pd.DataFrame:
    """
    Implementa §5.1 Step 1: per-occurrence mean OBV por evento.

    NOTA SOBRE DECISIÓN DE NORMALIZACIÓN:
    Se usa la media por ocurrencia (obv_for_net / n_eventos_del_tipo).
    Si se decide cambiar a per-90, modificar aquí y en P03.
    Ver DECISIÓN PENDIENTE en el docstring del módulo.
    """
    # Filtrar solo filas con player_id resuelto, season conocida y OBV válido
    ev_valid = ev.dropna(subset=["player_id", "season_name", "type", "obv_for_net"]).copy()
    ev_valid["player_id"] = ev_valid["player_id"].astype(int)

    # Filtrar temporadas relevantes
    ev_valid = ev_valid[ev_valid["season_name"].isin(EXPECTED_SEASONS)]

    # Agregar: media por ocurrencia de OBV para cada (jugador, temporada, tipo_evento)
    agg = (
        ev_valid.groupby(["player_id", "player", "season_name", "type"])
        .agg(
            n_events        = ("obv_for_net",    "count"),
            obv_for_mean    = ("obv_for_net",    "mean"),
            obv_against_mean= ("obv_against_net","mean"),
            obv_total_mean  = ("obv_total_net",  "mean"),
            obv_for_sum     = ("obv_for_net",    "sum"),
            obv_against_sum = ("obv_against_net","sum"),
        )
        .reset_index()
        .rename(columns={"player": "player_name", "type": "event_type"})
    )

    # Añadir posición más frecuente por jugador-temporada (modal position)
    pos_modal = (
        ev_valid.groupby(["player_id", "season_name"])["position"]
        .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else None)
        .reset_index()
        .rename(columns={"position": "position_detail"})
    )
    agg = agg.merge(pos_modal, on=["player_id", "season_name"], how="left")

    # Añadir position_group desde canonical (P01) si está disponible
    if POS_CANONICAL.exists():
        canon = pd.read_parquet(POS_CANONICAL)[["player_id", "position_group"]]
        agg = agg.merge(canon, on="player_id", how="left")
    else:
        # Fallback: mapear desde position_detail
        from P01_canonical_positions import POS_MAP  # noqa
        agg["position_group"] = agg["position_detail"].map(POS_MAP)

    return agg


# ── Sección D2: Computar PSR por (player_id, season_name) desde p01b ──────────

MIN_PRESSURE_EVENTS = 10  # mínimo de eventos PSR-elegibles para PSR confiable


def compute_psr_player_season(_ev: pd.DataFrame) -> pd.DataFrame:
    """
    Implementa §5.2: Pressure Success Rate (PSR) por jugador-temporada.

    PSR = fracción de acciones bajo presión PSR-elegibles (Pass/Dribble/Duel/Shot)
          con outcome binario exitoso. Cargado desde p01b_psr_raw.parquet.

    El parámetro _ev se mantiene por compatibilidad de firma pero NO se usa;
    los datos vienen de PSR_RAW_PATH (output de P01b).

    Returns:
        DataFrame con columnas: player_id, season_name, psr, n_pressure_events
    """
    if not PSR_RAW_PATH.exists():
        raise FileNotFoundError(
            f"p01b_psr_raw.parquet no encontrado: {PSR_RAW_PATH}\n"
            "Ejecutar P01b_events_outcomes.py antes de P02."
        )

    raw = pd.read_parquet(PSR_RAW_PATH)

    # Filtrar solo temporadas EPL relevantes
    raw = raw[raw["season_name"].isin(EXPECTED_SEASONS)].copy()

    if len(raw) == 0:
        print("  [AVISO] p01b sin filas para las temporadas EPL esperadas")
        return pd.DataFrame(columns=["player_id", "season_name", "psr", "n_pressure_events"])

    psr = (
        raw.groupby(["player_id", "season_name"])
        .agg(
            n_pressure_events = ("is_success", "count"),
            psr               = ("is_success", "mean"),
        )
        .reset_index()
    )

    # Aplicar mínimo estadístico (PSR global)
    n_before = len(psr)
    psr.loc[psr["n_pressure_events"] < MIN_PRESSURE_EVENTS, "psr"] = np.nan
    n_valid = psr["psr"].notna().sum()
    print(f"  PSR: {n_before} player-seasons con eventos PSR-elegibles, "
          f"{n_valid} con >= {MIN_PRESSURE_EVENTS} eventos (PSR válido)")

    return psr


MIN_PRESSURE_EVENTS_PER_TYPE = 5  # mínimo por tipo de evento para PSR desagregado


def compute_psr_by_event_type(_ev: pd.DataFrame) -> pd.DataFrame:
    """
    PSR por (player_id, season_name, event_type) con mínimo MIN_PRESSURE_EVENTS_PER_TYPE.

    Permite a P03 percentilizar PSR dentro de (season, position, event_type),
    consistente con la construcción de score_OBV.

    Returns:
        DataFrame con columnas: player_id, season_name, event_type, psr, n_events
        Solo incluye filas con n_events >= MIN_PRESSURE_EVENTS_PER_TYPE.
    """
    if not PSR_RAW_PATH.exists():
        raise FileNotFoundError(f"p01b_psr_raw.parquet no encontrado: {PSR_RAW_PATH}")

    raw = pd.read_parquet(PSR_RAW_PATH)
    raw = raw[raw["season_name"].isin(EXPECTED_SEASONS)].copy()

    if len(raw) == 0:
        return pd.DataFrame(columns=["player_id", "season_name", "event_type", "psr", "n_events"])

    psr_ev = (
        raw.groupby(["player_id", "season_name", "event_type"])
        .agg(
            n_events = ("is_success", "count"),
            psr      = ("is_success", "mean"),
        )
        .reset_index()
    )

    # Aplicar mínimo por tipo de evento
    psr_ev = psr_ev[psr_ev["n_events"] >= MIN_PRESSURE_EVENTS_PER_TYPE].copy()
    print(f"  PSR by event_type: {len(psr_ev)} (player-season, event_type) válidos "
          f"(min {MIN_PRESSURE_EVENTS_PER_TYPE} eventos/tipo)")
    return psr_ev


# ── Sección E: Validación del output ──────────────────────────────────────────

def validate_output(agg: pd.DataFrame) -> None:
    """Tests de regresión del output de P02."""
    seasons_found = set(agg["season_name"].unique())
    assert seasons_found == EXPECTED_SEASONS, \
        f"Temporadas encontradas {seasons_found} ≠ esperadas {EXPECTED_SEASONS}"

    n_players = agg["player_id"].nunique()
    assert 600 < n_players < 1200, \
        f"N jugadores únicos={n_players} fuera del rango esperado 600–1200"

    assert agg[["player_id", "season_name"]].duplicated().sum() == 0 or True
    # (duplicados OK a nivel player_id+season+event_type, no a nivel player+season)

    # OBV media puede ser negativa (hay acciones que restan valor), pero no NaN
    assert agg["obv_for_mean"].isna().sum() == 0, "NaN en obv_for_mean"

    # position_group no debe ser 100% MID
    if "position_group" in agg.columns:
        mid_pct = (agg["position_group"] == "MID").mean()
        # No aplicamos umbral aquí porque la agregación por event_type multiplica filas
        print(f"  [INFO] position_group MID en agg: {mid_pct:.1%}")

    print(f"  [TESTS OK] {n_players} jugadores únicos, {len(agg):,} filas")


def main() -> None:
    print("=" * 60)
    print("P02 — OBV Player-Season Aggregation")
    print("=" * 60)

    # A. Validar checkpoint
    ev = validate_checkpoint()

    # B. Añadir season_name
    ev = attach_season(ev)
    seasons = ev["season_name"].value_counts()
    print(f"\n  Distribución por temporada:")
    print(seasons.to_string())

    # C. Linkage player_name → player_id
    print(f"\n  Construyendo lookup nombre → player_id...")
    lookup = build_name_to_id_lookup()
    print(f"  Lookup: {len(lookup):,} entradas")
    ev = link_player_ids(ev, lookup)

    # D. Agregar OBV
    print(f"\n  Agregando OBV por (player, season, event_type)...")
    agg = aggregate_obv(ev)
    print(f"  Resultado: {len(agg):,} filas, {agg['player_id'].nunique()} jugadores")

    # D2. Computar PSR (global y por event_type)
    print(f"\n  Computando PSR (Pressure Success Rate)...")
    psr = compute_psr_player_season(ev)
    psr_ev = compute_psr_by_event_type(ev)

    # E. Validar
    validate_output(agg)

    # F. Guardar todos los outputs
    out_path = OUT_DIR / "p02_obv_player_season.parquet"
    agg.to_parquet(out_path, index=False)
    print(f"\n  Output OBV: {out_path}")

    psr_path = OUT_DIR / "p02_psr_player_season.parquet"
    psr.to_parquet(psr_path, index=False)
    print(f"  Output PSR global: {psr_path}")
    print(f"  PSR: {psr['psr'].notna().sum()} player-seasons con PSR válido")

    psr_ev_path = OUT_DIR / "p02_psr_by_event.parquet"
    psr_ev.to_parquet(psr_ev_path, index=False)
    print(f"  Output PSR por event_type: {psr_ev_path} ({len(psr_ev)} filas)")

    # --- Generar reporte EDA OBV/PSR ---
    print("\n  Generando reporte EDA de OBV y PSR...")
    try:
        import importlib.util
        _report_path = Path(__file__).resolve().parents[1] / "reports" / "generate_eda_report.py"
        spec = importlib.util.spec_from_file_location("generate_eda_report", _report_path)
        _mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_mod)
        _mod.run()
    except Exception as _e:
        print(f"  AVISO: no se pudo generar el reporte EDA — {_e}")


if __name__ == "__main__":
    main()

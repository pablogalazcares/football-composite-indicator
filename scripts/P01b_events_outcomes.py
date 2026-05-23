"""
P01b — PSR Raw Extraction from StatsBomb JSON
==============================================
Objetivo:
    Extraer outcomes binarios para acciones bajo presión desde los archivos JSON
    raw de StatsBomb Legacy (EPL 2022/23 y 2023/24), produciendo la materia prima
    para el cómputo del PSR en P02.

    Evento set PSR (canónico, §5.2):
      - Pass   : éxito ↔ outcome ausente (pase completado)
      - Dribble: éxito ↔ outcome == "Complete"
      - Duel   : éxito ↔ outcome ∈ {Won, Success In Play, Success Out}
      - Shot   : éxito ↔ outcome ≠ "Blocked"

    Carry EXCLUIDO: StatsBomb no registra campo outcome para carries.
    Duels sin outcome: excluidos del cómputo (no asignables a éxito/fracaso).

Upstream:
    data/raw/statsbomb_legacy/events/data/events/*.json   (580 archivos)
    data/processed/statsbomb_legacy/df_matches.csv        (match_id → season)

Output:
    outputs/pipeline/p01b_psr_raw.parquet
    Columnas: player_id, season_name, event_type, is_success
    Una fila por acción bajo presión PSR-elegible con outcome resolvible.

Season ID mapping:
    235 → "2022/2023"  (2022-08-05 a 2023-05-28)
    281 → "2023/2024"  (2023-08-11 a 2024-05-19)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]

EVENTS_DIR  = REPO / "data/raw/statsbomb_legacy/events/data/events"
MATCHES_CSV = REPO / "data/processed/statsbomb_legacy/df_matches.csv"
OUT_DIR     = REPO / "outputs/pipeline"

SEASON_MAP = {235: "2022/2023", 281: "2023/2024"}

PSR_EVENT_TYPES = {"Pass", "Dribble", "Duel", "Shot"}

DUEL_SUCCESS = {"Won", "Success In Play", "Success Out"}
DUEL_FAILURE = {"Lost", "Lost In Play", "Lost Out"}


def _outcome_name(event: dict, etype: str) -> str | None:
    """Extrae el nombre del outcome del subdict de tipo de evento."""
    sub = event.get(etype.lower(), {})
    outcome = sub.get("outcome", None)
    if outcome is None:
        return None
    return outcome.get("name", None) if isinstance(outcome, dict) else str(outcome)


def _is_success(etype: str, outcome: str | None) -> int | None:
    """
    Retorna 1 (éxito), 0 (fracaso) o None (no clasificable).
    None excluye la acción del cómputo PSR.
    """
    if etype == "Pass":
        # Pass completado: outcome ausente
        return 1 if outcome is None else 0

    if etype == "Dribble":
        if outcome == "Complete":
            return 1
        if outcome == "Incomplete":
            return 0
        return None  # outcome inesperado → excluir

    if etype == "Duel":
        if outcome in DUEL_SUCCESS:
            return 1
        if outcome in DUEL_FAILURE:
            return 0
        return None  # outcome ausente o no clasificable → excluir

    if etype == "Shot":
        # Éxito = disparo no bloqueado por defensa
        if outcome is None:
            return None  # shots siempre tienen outcome
        return 0 if outcome == "Blocked" else 1

    return None


def build_match_season_map() -> dict[int, str]:
    """Construye match_id → season_name desde df_matches.csv."""
    if not MATCHES_CSV.exists():
        raise FileNotFoundError(
            f"df_matches.csv no encontrado: {MATCHES_CSV}\n"
            "Necesario para asignar season_name a cada partido."
        )
    df = pd.read_csv(MATCHES_CSV, usecols=["match_id", "season"])
    mapping = {}
    unknown = set()
    for _, row in df.iterrows():
        season_name = SEASON_MAP.get(int(row["season"]))
        if season_name:
            mapping[int(row["match_id"])] = season_name
        else:
            unknown.add(int(row["season"]))
    if unknown:
        print(f"  [AVISO] Season IDs no reconocidos: {unknown} — matches excluidos")
    return mapping


def process_match_file(json_path: Path, season_name: str) -> list[dict]:
    """
    Lee un JSON de partido y extrae filas PSR para under_pressure events.
    Retorna lista de dicts {player_id, season_name, event_type, is_success}.
    """
    with open(json_path, encoding="utf-8") as f:
        events = json.load(f)

    rows: list[dict] = []
    for event in events:
        if not event.get("under_pressure"):
            continue
        etype = event.get("type", {}).get("name", "")
        if etype not in PSR_EVENT_TYPES:
            continue
        player_id = event.get("player", {}).get("id")
        if player_id is None:
            continue

        outcome = _outcome_name(event, etype)
        success = _is_success(etype, outcome)
        if success is None:
            continue  # acción no clasificable → excluir

        rows.append({
            "player_id":   int(player_id),
            "season_name": season_name,
            "event_type":  etype,
            "is_success":  success,
        })
    return rows


def main() -> None:
    print("=" * 60)
    print("P01b — PSR Raw Extraction from StatsBomb JSON")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # A. Construir mapa match_id → season_name
    print("\n  Construyendo mapa match_id → season_name...")
    match_season = build_match_season_map()
    print(f"  {len(match_season)} partidos mapeados a temporada")

    # B. Enumerar archivos JSON
    json_files = sorted(EVENTS_DIR.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No se encontraron JSON en {EVENTS_DIR}")
    print(f"  {len(json_files)} archivos JSON encontrados")

    # C. Procesar cada partido
    all_rows: list[dict] = []
    skipped = 0

    for i, json_path in enumerate(json_files, 1):
        match_id = int(json_path.stem)
        season_name = match_season.get(match_id)
        if season_name is None:
            skipped += 1
            continue

        rows = process_match_file(json_path, season_name)
        all_rows.extend(rows)

        if i % 100 == 0:
            print(f"  Procesados {i}/{len(json_files)} partidos "
                  f"({len(all_rows):,} eventos PSR acumulados)...")

    print(f"\n  Total: {len(json_files) - skipped} partidos procesados, "
          f"{skipped} saltados (season desconocida)")
    print(f"  Eventos PSR raw: {len(all_rows):,}")

    # D. Construir DataFrame y guardar
    raw = pd.DataFrame(all_rows)

    # Resumen por tipo de evento y temporada
    print("\n  Distribución por event_type y season_name:")
    summary = (
        raw.groupby(["season_name", "event_type"])
        .agg(n_events=("is_success", "count"),
             psr_mean=("is_success", "mean"))
        .round(3)
    )
    print(summary.to_string())

    # Resumen por jugador-temporada
    ps_counts = raw.groupby(["player_id", "season_name"]).size()
    print(f"\n  Player-seasons con >= 1 evento PSR: {len(ps_counts)}")
    print(f"  Mediana de eventos PSR por player-season: {ps_counts.median():.0f}")
    print(f"  Player-seasons con >= 10 eventos: {(ps_counts >= 10).sum()}")

    out_path = OUT_DIR / "p01b_psr_raw.parquet"
    raw.to_parquet(out_path, index=False)
    print(f"\n  Output: {out_path}")
    print(f"  Columnas: {list(raw.columns)}")
    print(f"  Filas: {len(raw):,}")


if __name__ == "__main__":
    main()

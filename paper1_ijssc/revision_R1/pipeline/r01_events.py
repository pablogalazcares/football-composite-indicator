"""R01 — Extracción mínima de eventos desde el JSON crudo de StatsBomb (REQ-D1, D3, R6).

Lee `events/{match_id}.json` de cada temporada configurada y escribe en la CACHÉ (fuera del repo, D-17):
  events_{label}.parquet     eventos con jugador: ids, tiempo, tipo, posición, presión, OBV, resultado, tarjetas
  lineups_sxi_{label}.parquet titulares por partido (Starting XI)
  subs_{label}.parquet       sustituciones (sale, entra, tiempo)
  period_end_{label}.parquet fin de cada período por partido (Half End)
En `outputs/r1/` (agregados, versionables) escribe:
  coverage_by_club_season.csv  partidos cubiertos por club y temporada
  ingest_manifest.json         partidos indexados, con eventos, faltantes, conteos
Soporta los dos formatos de Starting XI: anidado (A/B) y aplanado con claves 'player.id' (C).
"""
from __future__ import annotations

import json
from collections import defaultdict

import pandas as pd
from loguru import logger

from common import Config, load_config

PSR_TYPES = {"Pass", "Dribble", "Duel", "Shot"}
OUTCOME_KEY = {"Pass": "pass", "Dribble": "dribble", "Duel": "duel", "Shot": "shot", "Clearance": "clearance",
               "Interception": "interception", "Goal Keeper": "goalkeeper", "50/50": "50_50"}


def _ts_seconds(ts: str | None) -> float | None:
    if not ts:
        return None
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _sxi_rows(e: dict, match_id: int) -> list[dict]:
    rows = []
    for p in (e.get("tactics") or {}).get("lineup", []):
        if "player" in p:  # formato anidado
            pid, pname, pos = p["player"]["id"], p["player"].get("name"), (p.get("position") or {}).get("name")
        else:  # formato aplanado (Hudl 2024/25)
            pid, pname, pos = p.get("player.id"), p.get("player.name"), p.get("position.name")
        rows.append({"match_id": match_id, "team_id": e["team"]["id"], "team": e["team"]["name"],
                     "player_id": int(pid), "player_name": pname, "position": pos})
    return rows


def extract_match(events: list[dict], match_id: int) -> tuple[list, list, list, list]:
    ev_rows, sxi, subs, pend = [], [], [], []
    for e in events:
        et = (e.get("type") or {}).get("name")
        team = e.get("team") or {}
        if et == "Starting XI":
            sxi.extend(_sxi_rows(e, match_id))
            continue
        if et == "Half End":
            pend.append({"match_id": match_id, "period": e["period"], "end_s": _ts_seconds(e.get("timestamp")),
                         "end_minute": e.get("minute"), "end_second": e.get("second")})
            continue
        p = e.get("player")
        if not p:
            continue
        if et == "Substitution":
            rep = (e.get("substitution") or {}).get("replacement") or {}
            subs.append({"match_id": match_id, "team_id": team.get("id"), "period": e["period"],
                         "t_s": _ts_seconds(e.get("timestamp")), "off_id": p["id"], "on_id": rep.get("id"),
                         "on_name": rep.get("name")})
        sub = e.get(OUTCOME_KEY.get(et, ""), {}) if et in OUTCOME_KEY else {}
        card = ((e.get("bad_behaviour") or {}).get("card") or (e.get("foul_committed") or {}).get("card") or {})
        ev_rows.append({
            "match_id": match_id, "period": e.get("period"), "t_s": _ts_seconds(e.get("timestamp")),
            "minute": e.get("minute"), "second": e.get("second"), "type": et,
            "team_id": team.get("id"), "team": team.get("name"),
            "player_id": int(p["id"]), "player_name": p.get("name"), "nickname": p.get("nickname"),
            "position": (e.get("position") or {}).get("name"),
            "under_pressure": bool(e.get("under_pressure", False)),
            "obv_for_net": e.get("obv_for_net"), "obv_against_net": e.get("obv_against_net"),
            "obv_total_net": e.get("obv_total_net"),
            "outcome": ((sub or {}).get("outcome") or {}).get("name") if isinstance(sub, dict) else None,
            "duel_type": ((e.get("duel") or {}).get("type") or {}).get("name"),
            "card": card.get("name"),
        })
    return ev_rows, sxi, subs, pend


def run(cfg: Config) -> dict:
    manifest = {}
    cov_rows = []
    for s in cfg.seasons:
        idx = json.loads((cfg.paths["raw_matches"] / f"{s.season_id}.json").read_text(encoding="utf-8"))
        ev_all, sxi_all, subs_all, pend_all = [], [], [], []
        missing = []
        teams_by_match = {}
        for m in idx:
            mid = m["match_id"]
            home = m["home_team"]["home_team_name"] if isinstance(m.get("home_team"), dict) else m.get("home_team")
            away = m["away_team"]["away_team_name"] if isinstance(m.get("away_team"), dict) else m.get("away_team")
            teams_by_match[mid] = (home, away)
            f = cfg.paths["raw_events"] / f"{mid}.json"
            if not f.exists():
                missing.append(mid)
                continue
            ev, sxi, subs, pend = extract_match(json.loads(f.read_text(encoding="utf-8")), mid)
            ev_all += ev
            sxi_all += sxi
            subs_all += subs
            pend_all += pend
        ev_df = pd.DataFrame(ev_all)
        ev_df["season"] = s.label
        sxi_df = pd.DataFrame(sxi_all).assign(season=s.label)
        subs_df = pd.DataFrame(subs_all).assign(season=s.label)
        pend_df = (pd.DataFrame(pend_all).groupby(["match_id", "period"], as_index=False)
                   .agg(end_s=("end_s", "max"), end_minute=("end_minute", "max")).assign(season=s.label))
        ev_df.to_parquet(cfg.cache(f"events_{s.label}.parquet"), index=False)
        sxi_df.to_parquet(cfg.cache(f"lineups_sxi_{s.label}.parquet"), index=False)
        subs_df.to_parquet(cfg.cache(f"subs_{s.label}.parquet"), index=False)
        pend_df.to_parquet(cfg.cache(f"period_end_{s.label}.parquet"), index=False)
        # cobertura por club: partidos con eventos
        played = set(ev_df.match_id.unique())
        cnt = defaultdict(int)
        for mid in played:
            for t in teams_by_match.get(mid, ()):
                cnt[t] += 1
        # nombres de club desde eventos (más fiables que el índice)
        team_names = ev_df.groupby("match_id")["team"].unique()
        cnt_ev = defaultdict(int)
        for mid, tlist in team_names.items():
            for t in tlist:
                cnt_ev[t] += 1
        for t, n in cnt_ev.items():
            cov_rows.append({"season": s.label, "team": t, "matches_covered": n})
        manifest[s.label] = {
            "season_id": s.season_id, "matches_indexed": len(idx), "matches_with_events": len(played),
            "missing_event_files": missing, "n_events_with_player": int(len(ev_df)),
            "n_players": int(ev_df.player_id.nunique()), "n_teams": int(len(cnt_ev)),
            "obv_non_null_share": float(ev_df.obv_for_net.notna().mean()),
        }
        logger.info(f"{s.label}: {len(played)} partidos, {len(ev_df):,} eventos, faltan {len(missing)}")
    cov = pd.DataFrame(cov_rows).sort_values(["season", "team"])
    cov.to_csv(cfg.out("coverage_by_club_season.csv"), index=False)
    cfg.out("ingest_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    print(json.dumps(run(load_config()), indent=2))

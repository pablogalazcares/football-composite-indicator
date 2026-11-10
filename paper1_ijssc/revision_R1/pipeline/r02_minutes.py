"""R02 — Minutos jugados exactos por jugador-partido (REQ-D2, D3).

Fuente principal en TODAS las temporadas: eventos (Starting XI, Substitution, tarjeta roja o segunda amarilla en
Bad Behaviour / Foul Committed, Half End). El tiempo se mide en segundos transcurridos dentro de cada período
(`timestamp` de StatsBomb), incluido el descuento. No se descuentan las salidas temporales (Player Off/On).

Validación (A/B): los lineups de StatsBomb traen `positions` con from/to en reloj de partido (mm:ss acumulado), que
se convierten a tiempo transcurrido por período con offsets {1: 0, 2: 45, 3: 90, 4: 105} minutos.

Salidas (outputs/r1, agregados):
  minutes_player_match.parquet   minutos por (temporada, partido, jugador)
  minutes_player_season.parquet  minutos cubiertos, partidos jugados, club(es), minutos cubiertos del club, share
  minutes_validation.json        métricas de la validación contra lineups
y results/minutes_validation.md.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr

from common import Config, load_config

RED = {"Red Card", "Second Yellow"}
CLOCK_OFFSET_S = {1: 0, 2: 45 * 60, 3: 90 * 60, 4: 105 * 60}


def match_minutes(ev: pd.DataFrame, sxi: pd.DataFrame, subs: pd.DataFrame, pend: pd.DataFrame) -> pd.DataFrame:
    """Minutos por jugador y partido para una temporada."""
    ends = pend.pivot(index="match_id", columns="period", values="end_s")
    periods = [p for p in ends.columns if p in (1, 2, 3, 4)]
    reds = ev[ev.card.isin(RED)][["match_id", "player_id", "period", "t_s"]]
    team_of = ev.groupby(["match_id", "player_id"])["team"].agg(lambda x: x.mode().iloc[0])
    rows = []
    sub_by_match = {m: g for m, g in subs.groupby("match_id")}
    red_by_match = {m: g for m, g in reds.groupby("match_id")}
    for mid, g in sxi.groupby("match_id"):
        if mid not in ends.index:
            continue
        pe = {p: ends.loc[mid, p] for p in periods if pd.notna(ends.loc[mid, p])}
        last_p = max(pe)
        start = {int(pid): (1, 0.0) for pid in g.player_id}
        stop: dict[int, tuple[int, float]] = {}
        sm = sub_by_match.get(mid, pd.DataFrame(columns=subs.columns))
        for r in sm.itertuples():
            stop.setdefault(int(r.off_id), (int(r.period), float(r.t_s)))
            if pd.notna(r.on_id):
                start.setdefault(int(r.on_id), (int(r.period), float(r.t_s)))
        rm = red_by_match.get(mid)
        if rm is not None:
            for r in rm.itertuples():
                pid = int(r.player_id)
                if pid in start:
                    prev = stop.get(pid)
                    cand = (int(r.period), float(r.t_s))
                    if prev is None or cand < prev:
                        stop[pid] = cand
        for pid, (p0, t0) in start.items():
            p1, t1 = stop.get(pid, (last_p, pe[last_p]))
            secs = 0.0
            for p in sorted(pe):
                if p < p0 or p > p1:
                    continue
                a = t0 if p == p0 else 0.0
                b = t1 if p == p1 else pe[p]
                secs += max(0.0, b - a)
            rows.append({"match_id": mid, "player_id": pid, "minutes": secs / 60.0,
                         "started": pid in set(g.player_id), "team": team_of.get((mid, pid), None)})
    out = pd.DataFrame(rows)
    # equipo para quienes no tocaron el balón (sin eventos): del Starting XI o de la sustitución
    sxi_team = sxi.set_index(["match_id", "player_id"])["team"]
    miss = out.team.isna()
    out.loc[miss, "team"] = [sxi_team.get((m, p)) for m, p in zip(out.match_id[miss], out.player_id[miss])]
    return out


def lineup_minutes(cfg: Config, match_ids: list[int], pend: pd.DataFrame) -> pd.DataFrame:
    """Minutos según los lineups (solo existen en A/B)."""
    ends = pend.pivot(index="match_id", columns="period", values="end_s")
    rows = []
    for mid in match_ids:
        f = cfg.paths["raw_lineups"] / f"{mid}.json"
        if not f.exists():
            continue
        last_p = int(ends.loc[mid].dropna().index.max())
        for team in json.loads(f.read_text(encoding="utf-8")):
            for p in team["lineup"]:
                secs = 0.0
                for pos in p.get("positions") or []:
                    fp, tp = int(pos["from_period"]), pos.get("to_period")
                    f_s = _clock_to_elapsed(pos["from"], fp)
                    if pos.get("to") is None:
                        tp, t_s = last_p, ends.loc[mid, last_p]
                    else:
                        tp = int(tp)
                        t_s = _clock_to_elapsed(pos["to"], tp)
                    for per in range(fp, tp + 1):
                        if per not in ends.columns or pd.isna(ends.loc[mid, per]):
                            continue
                        a = f_s if per == fp else 0.0
                        b = t_s if per == tp else ends.loc[mid, per]
                        secs += max(0.0, b - a)
                if p.get("positions"):
                    rows.append({"match_id": mid, "player_id": int(p["player_id"]), "minutes_lineup": secs / 60.0,
                                 "nickname": p.get("player_nickname"), "birth_date": p.get("birth_date")})
    return pd.DataFrame(rows)


def _clock_to_elapsed(clock: str, period: int) -> float:
    h, m, s = clock.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s) - CLOCK_OFFSET_S.get(period, 0)


def run(cfg: Config) -> dict:
    cov = pd.read_csv(cfg.out("coverage_by_club_season.csv"))
    pm_all, val = [], {}
    lineup_info = []
    for s in cfg.seasons:
        ev = pd.read_parquet(cfg.cache(f"events_{s.label}.parquet"),
                             columns=["match_id", "player_id", "team", "card", "period", "t_s"])
        sxi = pd.read_parquet(cfg.cache(f"lineups_sxi_{s.label}.parquet"))
        subs = pd.read_parquet(cfg.cache(f"subs_{s.label}.parquet"))
        pend = pd.read_parquet(cfg.cache(f"period_end_{s.label}.parquet"))
        mm = match_minutes(ev, sxi, subs, pend).assign(season=s.label)
        pm_all.append(mm)
        lu = lineup_minutes(cfg, sorted(mm.match_id.unique()), pend)
        if len(lu):
            lineup_info.append(lu.assign(season=s.label))
            j = mm.merge(lu, on=["match_id", "player_id"], how="outer")
            both = j.dropna(subset=["minutes", "minutes_lineup"])
            err = (both.minutes - both.minutes_lineup).abs()
            tot = j.groupby("player_id")[["minutes", "minutes_lineup"]].sum(min_count=1).dropna()
            val[s.label] = {
                "player_matches_events": int(mm.shape[0]), "player_matches_lineups": int(lu.shape[0]),
                "only_in_events": int(j.minutes_lineup.isna().sum()), "only_in_lineups": int(j.minutes.isna().sum()),
                "median_abs_err_min": float(err.median()), "p95_abs_err_min": float(err.quantile(0.95)),
                "share_abs_err_gt_flag": float((err > cfg.raw["validation"]["minutes_flag_discrepancy"]).mean()),
                "season_total_spearman": float(spearmanr(tot.minutes, tot.minutes_lineup)[0]),
                "season_total_mean_diff_min": float((tot.minutes - tot.minutes_lineup).mean()),
            }
            big = both[err > cfg.raw["validation"]["minutes_flag_discrepancy"]]
            big.assign(abs_err=err[err > cfg.raw["validation"]["minutes_flag_discrepancy"]]).to_csv(
                cfg.out(f"minutes_discrepancies_{s.label}.csv"), index=False)
    pm = pd.concat(pm_all, ignore_index=True)
    pm.to_parquet(cfg.out("minutes_player_match.parquet"), index=False)
    if lineup_info:
        pd.concat(lineup_info).to_parquet(cfg.cache("lineup_player_info.parquet"), index=False)
    # agregado por jugador-temporada
    club_min = (pm.groupby(["season", "team", "match_id"]).size().reset_index()[["season", "team", "match_id"]])
    pend_all = pd.concat([pd.read_parquet(cfg.cache(f"period_end_{s.label}.parquet")) for s in cfg.seasons])
    mlen = pend_all.groupby(["season", "match_id"])["end_s"].sum().div(60).rename("match_minutes").reset_index()
    club_min = club_min.merge(mlen, on=["season", "match_id"]).groupby(["season", "team"], as_index=False).agg(
        club_matches_covered=("match_id", "nunique"), club_minutes_covered=("match_minutes", "sum"))
    by_team = pm.groupby(["season", "player_id", "team"], as_index=False).agg(minutes=("minutes", "sum"),
                                                                              matches=("match_id", "nunique"))
    by_team = by_team.merge(club_min, on=["season", "team"], how="left")
    by_team = by_team.sort_values(["season", "player_id", "minutes"], ascending=[True, True, False])
    agg = by_team.groupby(["season", "player_id"]).agg(
        minutes_covered=("minutes", "sum"), matches_played=("matches", "sum"),
        clubs=("team", lambda x: "|".join(x)), main_club=("team", "first"),
        club_matches_covered=("club_matches_covered", "first"),
        club_minutes_covered=("club_minutes_covered", "first")).reset_index()
    # share respecto del club principal; con varios clubes se usa la suma de minutos cubiertos de esos clubes
    multi = by_team.groupby(["season", "player_id"])["club_minutes_covered"].sum().rename("clubs_minutes_covered")
    agg = agg.merge(multi.reset_index(), on=["season", "player_id"])
    agg["share_club_minutes"] = agg.minutes_covered / agg.clubs_minutes_covered
    agg.to_parquet(cfg.out("minutes_player_season.parquet"), index=False)
    cfg.out("minutes_validation.json").write_text(json.dumps(val, indent=2))
    _report(cfg, val, agg, cov)
    return val


def _report(cfg: Config, val: dict, agg: pd.DataFrame, cov: pd.DataFrame) -> None:
    v = cfg.raw["validation"]
    lines = ["# Validación de minutos reconstruidos desde eventos (REQ-D2)", "",
             "Método: titulares por Starting XI, entradas y salidas por Substitution, salidas por roja o segunda "
             "amarilla, fin de período por Half End; tiempo transcurrido por período, incluido el descuento. "
             "Referencia: `positions` de los lineups de StatsBomb (solo existen en 2022/23 y 2023/24).", "",
             f"Criterios: error absoluto mediano ≤ {v['minutes_median_abs_err_max']} min por jugador-partido; "
             f"Spearman de totales por temporada ≥ {v['minutes_season_total_rho_min']}.", "",
             "| Temporada | Jugador-partidos (eventos / lineups) | Solo en eventos | Solo en lineups | Error mediano (min) | "
             "Error p95 (min) | % > 10 min | ρ totales | Δ media total (min) | Cumple |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for k, x in val.items():
        ok = x["median_abs_err_min"] <= v["minutes_median_abs_err_max"] and \
            x["season_total_spearman"] >= v["minutes_season_total_rho_min"]
        lines.append(f"| {k} | {x['player_matches_events']} / {x['player_matches_lineups']} | {x['only_in_events']} | "
                     f"{x['only_in_lineups']} | {x['median_abs_err_min']:.2f} | {x['p95_abs_err_min']:.2f} | "
                     f"{100 * x['share_abs_err_gt_flag']:.2f} | {x['season_total_spearman']:.4f} | "
                     f"{x['season_total_mean_diff_min']:.1f} | {'sí' if ok else 'NO'} |")
    lines += ["", "La temporada 2024/25 no tiene lineups: sus minutos salen del mismo método validado aquí.", "",
              "Discrepancias > 10 min: `outputs/r1/minutes_discrepancies_<temporada>.csv`.", "",
              "## Exposición por cobertura del club", "",
              "| Temporada | Partidos cubiertos del club | Jugador-temporadas | ≥ 1.000 min | % ≥ 1.000 |", "|---|---|---|---|---|"]
    for (sea, cm), g in agg.groupby(["season", "club_matches_covered"]):
        n = len(g)
        n1 = int((g.minutes_covered >= cfg.raw["exposure"]["main_min_minutes"]).sum())
        lines.append(f"| {sea} | {int(cm)} | {n} | {n1} | {100 * n1 / n:.1f} |")
    cfg.res("minutes_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    print(json.dumps(run(load_config()), indent=2))

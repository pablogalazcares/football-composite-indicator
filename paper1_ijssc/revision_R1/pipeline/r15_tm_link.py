"""R15 — Enlace StatsBomb ↔ Transfermarkt y valor de mercado de fin de temporada (PREREGISTRO_ADENDA_1 §2).

Regla: misma fecha de nacimiento; Jaccard de tokens normalizados del nombre ≥ 0,25 (máximo entre el nombre de
StatsBomb y el nombre de uso común); en empate gana la coincidencia de club. Candidatos: `tm_squad` (todas las ligas) de
las temporadas configuradas.
Valor de fin de temporada t: última valoración del historial con fecha ≤ 30/06 del fin de t; si el jugador no está en el
historial, el `market_value_eur` de su fila de `tm_squad` en t.
Salidas: outputs/r1/tm_link.parquet, outputs/r1/mv_panel.parquet, outputs/r1/tm_link_report.json,
         results/tm_link_review.md (muestra de 30 pares).
"""
from __future__ import annotations

import json
import sqlite3
import unicodedata

import numpy as np
import pandas as pd

from common import Config, load_config


def norm_tokens(s: str | None) -> set[str]:
    if not s:
        return set()
    s = "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c)).lower()
    for ch in "-'.’":
        s = s.replace(ch, " ")
    return {t for t in s.split() if t}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def run(cfg: Config) -> dict:
    mvc = cfg.raw["market_value"]
    con = sqlite3.connect(str(cfg.paths["tm_sqlite"].resolve()))
    seasons = ",".join(str(x) for x in mvc["tm_seasons"])
    tm = pd.read_sql(f"""select s.season_id, s.club_id, s.player_id as tm_id, s.player_name as tm_name, s.dob,
                                s.market_value_eur, c.club_name, c.comp_code
                         from tm_squad s left join tm_comp_season_club c
                           on s.club_id = c.club_id and s.season_id = c.season_id
                         where s.season_id in ({seasons})""", con)
    con.close()
    tm["dob"] = pd.to_datetime(tm.dob, errors="coerce")
    ps = pd.read_parquet(cfg.out("players_season.parquet"))
    sb = ps.drop_duplicates("player_id")[["player_id", "player_name", "display_name", "birth_date"]].copy()
    clubs = ps.groupby("player_id").clubs.agg(lambda x: set("|".join(x).split("|")))
    sb["clubs"] = sb.player_id.map(clubs)
    tm_by_dob = {d: g for d, g in tm.dropna(subset=["dob"]).groupby("dob")}
    rows = []
    for r in sb.itertuples():
        if pd.isna(r.birth_date):
            rows.append({"player_id": r.player_id, "tm_id": np.nan, "score": 0.0, "status": "no_birth_date"})
            continue
        cand = tm_by_dob.get(pd.Timestamp(r.birth_date))
        if cand is None:
            rows.append({"player_id": r.player_id, "tm_id": np.nan, "score": 0.0, "status": "no_dob_candidate"})
            continue
        names = [norm_tokens(r.player_name), norm_tokens(r.display_name)]
        best = []
        for tid, g in cand.groupby("tm_id"):
            sc = max(jaccard(n, norm_tokens(g.tm_name.iloc[0])) for n in names)
            club_hit = any(any(cn and (cn.lower() in str(x).lower() or str(x).lower() in cn.lower())
                               for x in r.clubs) for cn in g.club_name.dropna().unique())
            best.append((sc, club_hit, tid, g.tm_name.iloc[0]))
        best.sort(key=lambda x: (x[0], x[1]), reverse=True)
        sc, club_hit, tid, tname = best[0]
        if sc >= float(mvc["name_min_jaccard"]):
            rows.append({"player_id": r.player_id, "tm_id": int(tid), "tm_name": tname, "score": sc,
                         "club_hit": club_hit, "n_candidates": len(best), "status": "linked"})
        else:
            rows.append({"player_id": r.player_id, "tm_id": np.nan, "tm_name": tname, "score": sc,
                         "status": "below_threshold"})
    link = pd.DataFrame(rows).merge(sb[["player_id", "player_name", "display_name", "birth_date"]], on="player_id")
    dup = link.dropna(subset=["tm_id"]).tm_id.duplicated(keep=False)
    link["tm_id_shared"] = False
    link.loc[link.dropna(subset=["tm_id"]).index[dup.values], "tm_id_shared"] = True
    link.to_parquet(cfg.out("tm_link.parquet"), index=False)

    # valor de fin de temporada
    hist = pd.read_parquet(cfg.paths["tm_history"].resolve(), columns=["pid", "fecha", "valor_eur"])
    hist = hist[hist.pid.isin(link.tm_id.dropna().astype(int))].sort_values(["pid", "fecha"])
    mv_rows = []
    linked = link.dropna(subset=["tm_id"])
    for lab, end in mvc["season_end"].items():
        end = pd.Timestamp(end)
        h = hist[hist.fecha <= end].groupby("pid").last()
        sq = tm[tm.season_id == mvc["tm_season_of"][lab]].dropna(subset=["market_value_eur"]) \
            .groupby("tm_id").market_value_eur.max()
        for r in linked.itertuples():
            tid = int(r.tm_id)
            if tid in h.index:
                mv_rows.append({"player_id": r.player_id, "season": lab, "mv_eur": float(h.loc[tid, "valor_eur"]),
                                "mv_date": h.loc[tid, "fecha"], "source": "history"})
            elif tid in sq.index:
                mv_rows.append({"player_id": r.player_id, "season": lab, "mv_eur": float(sq.loc[tid]),
                                "mv_date": end, "source": "squad"})
    mv = pd.DataFrame(mv_rows)
    mv.to_parquet(cfg.out("mv_panel.parquet"), index=False)
    rep = {"sb_players": int(len(sb)), "status": link.status.value_counts().to_dict(),
           "link_rate": float((link.status == "linked").mean()),
           "tm_id_shared_by_two_sb_players": int(link.tm_id_shared.sum()),
           "mv_by_season_source": mv.groupby(["season", "source"]).size().unstack(fill_value=0).to_dict("index"),
           "history_pids_available": int(hist.pid.nunique())}
    # tasa de enlace entre elegibles por temporada
    el = ps[ps.minutes_covered >= cfg.raw["exposure"]["main_min_minutes"]]
    rep["eligible_link_rate_by_season"] = el.merge(link[["player_id", "status"]], on="player_id").groupby("season") \
        .status.apply(lambda x: float((x == "linked").mean())).to_dict()
    cfg.out("tm_link_report.json").write_text(json.dumps(rep, indent=2, default=str))
    rng = np.random.default_rng(cfg.seed)
    smp = linked.iloc[rng.choice(len(linked), size=min(30, len(linked)), replace=False)]
    L = ["# Revisión manual del enlace StatsBomb ↔ Transfermarkt (muestra aleatoria de 30)", "",
         f"Enlazados: {rep['status'].get('linked', 0)} de {rep['sb_players']} ({100 * rep['link_rate']:.1f} %). "
         f"Estados: {rep['status']}. IDs de Transfermarkt compartidos por dos jugadores de StatsBomb: "
         f"{rep['tm_id_shared_by_two_sb_players']}.", "",
         "| StatsBomb | Uso común | Nacimiento | Transfermarkt | ID TM | Jaccard | Club coincide |", "|---|---|---|---|---|---|---|"]
    for r in smp.itertuples():
        L.append(f"| {r.player_name} | {r.display_name} | {pd.Timestamp(r.birth_date).date()} | {r.tm_name} | {int(r.tm_id)} | "
                 f"{r.score:.2f} | {'sí' if r.club_hit else 'no'} |")
    below = link[link.status == "below_threshold"].head(15)
    L += ["", "Bajo el umbral (no enlazados), primeros 15:", "", "| StatsBomb | Mejor candidato TM | Jaccard |", "|---|---|---|"]
    for r in below.itertuples():
        L.append(f"| {r.player_name} | {r.tm_name} | {r.score:.2f} |")
    cfg.res("tm_link_review.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return rep


if __name__ == "__main__":
    print(json.dumps(run(load_config()), indent=2, default=str))

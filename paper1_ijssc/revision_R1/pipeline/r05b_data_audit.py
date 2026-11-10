"""R05b — Reporte de auditoría de la capa de datos (T1.8, insumo de HC-1).

Solo conteos y descriptivos por temporada. NO calcula persistencia ni ninguna métrica entre temporadas
(el preregistro, G2, va antes de eso).
Salida: results/data_audit_R1.md y outputs/r1/data_audit.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import Config, load_config

SMAP = {"A": "2022/2023", "B": "2023/2024", "C": "2024/2025"}


def run(cfg: Config) -> dict:
    A: dict = {}
    L: list[str] = ["# Auditoría de la capa de datos R1 (T1.8)", "",
                    "Generado por `pipeline/r05b_data_audit.py`. Solo descriptivos por temporada; ninguna métrica "
                    "entre temporadas (se calculan después del preregistro).", ""]
    man = json.loads(cfg.out("ingest_manifest.json").read_text())
    cov = pd.read_csv(cfg.out("coverage_by_club_season.csv"))
    L += ["## 1. Cobertura", "", "| Temporada | Partidos indexados | Con eventos | Clubes por partidos cubiertos |",
          "|---|---|---|---|"]
    for s, m in man.items():
        dist = cov[cov.season == s].matches_covered.value_counts().sort_index().to_dict()
        L.append(f"| {s} ({SMAP[s]}) | {m['matches_indexed']} | {m['matches_with_events']} | "
                 + ", ".join(f"{v} clubes con {k}" for k, v in dist.items()) + " |")
        A[f"coverage_{s}"] = dist
    L += ["", "En 2024/25 la entrega trae 378 de los 380 partidos: los 2 faltantes no están en el índice.", ""]

    # 2. minutos
    val = json.loads(cfg.out("minutes_validation.json").read_text())
    ps = pd.read_parquet(cfg.out("players_season_full.parquet"))
    psf = pd.read_parquet(cfg.paths["submitted_pipeline"].parent.parent / "data/processed/modeling/player_season_features.parquet")
    psf = psf[psf.competition_name == "Premier League"].dropna(subset=["minutes_played"]).drop_duplicates()
    psf = psf.groupby(["player_id", "season_name"], as_index=False)["minutes_played"].sum()
    psf["season"] = psf.season_name.map({v: k for k, v in SMAP.items()})
    j = psf.merge(ps[["player_id", "season", "minutes_covered", "matches_played"]], on=["player_id", "season"])
    j["ratio"] = j.minutes_played / j.minutes_covered
    pairs = pd.read_parquet(cfg.paths["submitted_pipeline"] / "p04_training_pairs_1000min.parquet")
    w = ps.pivot(index="player_id", columns="season", values="minutes_covered")
    ok_true = ((w.reindex(pairs.player_id)["A"] >= 1000) & (w.reindex(pairs.player_id)["B"] >= 1000)).sum()
    A["minutes"] = {"validation": val, "psf_over_true_ratio_median": float(j.ratio.median()),
                    "psf_over_true_ratio_p90": float(j.ratio.quantile(0.9)),
                    "psf_ge1000_AB": {s: int((j[j.season == s].minutes_played >= 1000).sum()) for s in ("A", "B")},
                    "true_ge1000_AB": {s: int((j[j.season == s].minutes_covered >= 1000).sum()) for s in ("A", "B")},
                    "submitted_211_pairs_true_ge1000_both": int(ok_true)}
    L += ["## 2. Minutos", "",
          f"- **Reconstrucción validada contra lineups** (A/B): error mediano {val['A']['median_abs_err_min']:.2f} / "
          f"{val['B']['median_abs_err_min']:.2f} min por jugador-partido; ρ de totales {val['A']['season_total_spearman']:.4f} / "
          f"{val['B']['season_total_spearman']:.4f}. Detalle en `minutes_validation.md`.",
          "- **Hallazgo sobre la versión enviada.** Los minutos de `data/processed/modeling/player_season_features.parquet`, "
          "que usó P04 para el umbral de 1.000, están inflados frente a los reales:",
          f"  - razón mediana (enviado / real) {j.ratio.median():.2f} y p90 {j.ratio.quantile(0.9):.2f};",
          "  - el origen probable es la inferencia `max(minute) + 1` por partido "
          "(`src/data_processing/frames_processor.py`), que a un suplente le acredita casi el partido completo;",
          f"  - con minutos reales, ≥ 1.000 minutos tienen {A['minutes']['true_ge1000_AB']['A']} jugadores en A "
          f"(no {A['minutes']['psf_ge1000_AB']['A']}) y {A['minutes']['true_ge1000_AB']['B']} en B (no "
          f"{A['minutes']['psf_ge1000_AB']['B']});",
          f"  - de los 211 pares del paper, **solo {ok_true}** cumplen ≥ 1.000 minutos reales en ambas temporadas.",
          "  - Afecta la muestra de entrenamiento (N = 211) y todo lo que depende de ella. Debe declararse en la "
          "respuesta a M7.", ""]

    # 3. edad, posición, clubes, identidad
    stats = json.loads(cfg.out("players_season_stats.json").read_text())
    L += ["## 3. Jugadores", "", "| Temporada | Jugadores con minutos | Elegibles (≥ 1.000) | Sin edad (todos / elegibles) | "
          "Origen de la fecha de nacimiento | Con más de un club |", "|---|---|---|---|---|---|"]
    for s in cfg.labels:
        x = stats[s]
        L.append(f"| {s} | {x['players']} | {x['eligible_main']} | {x['missing_age_all']} / {x['missing_age_eligible']} | "
                 f"{x['birth_source']} | {x['multi_club']} |")
    pc = stats["position_changes"]
    L += ["", "Cambios de grupo posicional (modal por temporada):"]
    for k, v in pc.items():
        L.append(f"- {k}: {v['changed']} de {v['in_both']} jugadores presentes en ambas; {v['eligible_changed']} de "
                 f"{v['eligible_both']} elegibles en ambas.")
    L.append(f"- En A/B, el grupo de la temporada difiere del canónico de P01 en {stats['season_vs_canonical_differs_AB']} "
             "jugador-temporadas.")
    named = ps[ps.player_id.isin([3272, 6802, 4127, 307036, 48706, 23230])][
        ["season", "display_name", "minutes_covered", "matches_played", "main_club", "eligible_main", "score_obv_ar1"]]
    L += ["", "Casos señalados por los revisores o por el diagnóstico (capa nueva):", "",
          "| Temporada | Jugador | Minutos | Partidos | Club | Elegible | Score |", "|---|---|---|---|---|---|---|"]
    for r in named.sort_values(["display_name", "season"]).itertuples():
        L.append(f"| {r.season} | {r.display_name} | {r.minutes_covered:.0f} | {r.matches_played} | {r.main_club} | "
                 f"{'sí' if r.eligible_main else 'no'} | {r.score_obv_ar1:.1f} |")
    loans = ps[ps.clubs.str.contains(r"\|")][["season", "display_name", "clubs", "main_club", "minutes_covered"]]
    L += ["", f"Jugadores con más de un club en la temporada: {len(loans)}. El club principal es el de más minutos; la lista "
          "completa está en `players_season.parquet` (columna `clubs`).", ""]

    # 4. under_pressure y OBV por temporada
    rows, obv = [], []
    for s in cfg.labels:
        ev = pd.read_parquet(cfg.cache(f"events_{s}.parquet"), columns=["type", "under_pressure", "obv_for_net"])
        g = ev.groupby("type").agg(n=("type", "size"), up=("under_pressure", "mean"))
        for t in ["Pass", "Carry", "Dribble", "Duel", "Shot", "Clearance"]:
            if t in g.index:
                rows.append({"season": s, "type": t, "share_under_pressure": round(float(g.loc[t, "up"]), 3)})
        e2 = ev.dropna(subset=["obv_for_net"])
        dec = e2.obv_for_net.astype(str).str.split(".").str[-1].str.len().median()
        for t in ["Pass", "Carry", "Shot"]:
            x = e2[e2.type == t].obv_for_net
            obv.append({"season": s, "type": t, "mean": round(float(x.mean()), 5), "sd": round(float(x.std()), 5),
                        "median_decimals": float(dec)})
    upd = pd.DataFrame(rows).pivot(index="type", columns="season", values="share_under_pressure")
    obd = pd.DataFrame(obv)
    A["under_pressure_by_type"] = upd.to_dict()
    A["obv_for_by_type"] = obd.to_dict("records")
    L += ["## 4. Comparabilidad entre temporadas", "", "Proporción de eventos con `under_pressure` por tipo:", "",
          "| Tipo | " + " | ".join(upd.columns) + " |", "|---|" + "---|" * len(upd.columns)]
    for t, r in upd.iterrows():
        L.append(f"| {t} | " + " | ".join(f"{v:.3f}" for v in r.values) + " |")
    L += ["", "Duel, Dribble y Clearance llevan `under_pressure` en el 100 % de los casos, por definición de StatsBomb "
          "(Data Specification v1.1, Appendix 7).", "", "OBV FOR por tipo (media, DE, decimales de la entrega):", "",
          "| Temporada | Tipo | Media | DE | Decimales (mediana) |", "|---|---|---|---|---|"]
    for r in obd.itertuples():
        L.append(f"| {r.season} | {r.type} | {r.mean} | {r.sd} | {r.median_decimals:.0f} |")
    L += ["", "La entrega 2024/25 trae OBV redondeado a 4 decimales. Como el indicador usa percentiles dentro de la temporada, "
          "el redondeo y cualquier cambio de nivel del modelo OBV entre entregas afectan solo los empates; igual se declara.", ""]

    # 5. indicador
    chk = json.loads(cfg.out("indicator_checks.json").read_text())
    A["indicator_checks"] = chk
    L += ["## 5. Indicador", "",
          f"- **Modo legacy:** reproduce `score_OBV` de P03 en {chk['legacy_n_compared']} jugador-temporadas, con "
          f"diferencia máxima {chk['legacy_max_abs_diff_vs_p03']:.1e} y el mismo patrón de faltantes. Las medias por "
          f"tipo desde el crudo coinciden con p02 ({chk['raw_vs_p02_rows_matched']} filas, diferencia "
          f"{chk['raw_vs_p02_max_abs_diff_for']}).",
          f"- **Pesos AR1** re-estimados desde crudo solo con A→B: reproducen las constantes de P03 (diferencia máxima "
          f"{chk['ar1_reestimated_max_abs_diff_vs_constants']:.1e}; mismos conjuntos de eventos).",
          f"- **Modo revisión frente a legacy en A/B** (mismo jugador-temporada): Spearman "
          f"{chk['revision_vs_legacy_AB_spearman']:.3f}, diferencia máxima {chk['revision_vs_legacy_AB_max_abs_diff']:.1f} "
          "puntos. La causa es la posición de la temporada en vez de la canónica (cambia el grupo del percentil).",
          f"- Jugador-temporadas con score en modo revisión: {chk['revision_n_by_season']}.", ""]

    # 6. muestras
    summ = json.loads(cfg.out("samples_summary.json").read_text())
    L += ["## 6. Muestras (D-04: ≥ 1.000 minutos reales)", "", "| Transición | Con score en t | Elegibles en t | "
          "Elegibles en ambas | Elegibles en t ausentes en t+k | Presentes pero no elegibles en t+k |",
          "|---|---|---|---|---|---|"]
    for k, v in summ["transitions"].items():
        L.append(f"| {k} | {v['scored_t']} | {v['eligible_t']} | {v['eligible_both']} | {v['eligible_t_absent_tk']} | "
                 f"{v['eligible_t_present_not_eligible_tk']} |")
    L += ["", f"Elegibles en las tres temporadas: {summ['same_set_all_seasons']}. Universo del panel (base 2024/25, "
          f"elegibles): {summ['panel_universe']}.", "", "Elegibles por temporada y posición: "
          + json.dumps(summ["eligible_main_by_season_pos"]), "",
          "## 7. Criterios de bloqueo de HC-1", ""]
    crit = {"REQ-D2 (minutos)": all(val[s]["median_abs_err_min"] <= 2 and val[s]["season_total_spearman"] >= 0.99
                                    for s in ("A", "B")),
            "REQ-D4 (≤ 1 % de elegibles sin edad)": all(stats[s]["missing_age_eligible"] <= 0.01 * stats[s]["eligible_main"]
                                                          for s in cfg.labels),
            "REQ-C1 (legacy = P03)": chk["legacy_max_abs_diff_vs_p03"] <= 1e-9 and chk["legacy_nan_pattern_equal"]}
    for k, v in crit.items():
        L.append(f"- {k}: {'cumple' if v else 'NO CUMPLE: bloquea'}")
    A["hc1_criteria"] = crit
    cfg.res("data_audit_R1.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    cfg.out("data_audit.json").write_text(json.dumps(A, indent=2, default=str))
    return A


if __name__ == "__main__":
    out = run(load_config())
    print(json.dumps(out["hc1_criteria"], indent=2), json.dumps(out["minutes"], indent=1, default=str)[:1500])

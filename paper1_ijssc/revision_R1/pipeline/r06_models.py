"""R06 — Modelos preespecificados M0–M3 ajustados SOLO en A→B (PREREGISTRO P2, P3, P7; SPEC §5.4).

M0: ŝ = s_t
M1: AR(1) + posición (OLS)
M2: M1 + (edad−c) + (edad−c)²  (c = 26)
M3: M2 + posición × (edad−c, (edad−c)²)   (exploratorio)
Referencia de posición: DEF (GK fuera, D-03; con GK en sensibilidad, GK como un grupo más).

Funciones reutilizadas por r07–r12 y r14. `run` escribe outputs/r1/model_fits.json y predictions.parquet.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from common import OUTFIELD, Config, load_config

MODELS = ("M0", "M1", "M2", "M3")


@dataclass
class Fitted:
    name: str
    groups: tuple[str, ...]
    center: float
    intercept: float = 0.0
    coef: np.ndarray | None = None
    names: list[str] | None = None

    def design(self, s, age, pos) -> np.ndarray:
        s, a, pos = np.asarray(s, float), np.asarray(age, float) - self.center, np.asarray(pos)
        cols, names = [s], ["s_t"]
        dums = [g for g in self.groups[1:]]
        for g in dums:
            cols.append((pos == g).astype(float))
            names.append(f"pos_{g}")
        if self.name in ("M2", "M3"):
            cols += [a, a ** 2]
            names += ["age_c", "age_c2"]
        if self.name == "M3":
            for g in dums:
                d = (pos == g).astype(float)
                cols += [d * a, d * a ** 2]
                names += [f"age_c_x_{g}", f"age_c2_x_{g}"]
        self.names = names
        return np.column_stack(cols)

    def fit(self, s, age, pos, y) -> "Fitted":
        if self.name == "M0":
            return self
        m = LinearRegression().fit(self.design(s, age, pos), np.asarray(y, float))
        self.intercept, self.coef = float(m.intercept_), m.coef_.copy()
        return self

    def predict(self, s, age, pos) -> np.ndarray:
        if self.name == "M0":
            return np.asarray(s, float)
        return self.intercept + self.design(s, age, pos) @ self.coef

    def project(self, s, age, pos, k: int) -> np.ndarray:
        """Proyección recursiva de k pasos, con la edad + 1 por paso y la posición fija en la de origen."""
        cur, a = np.asarray(s, float), np.asarray(age, float)
        for _ in range(k):
            cur = self.predict(cur, a, pos)
            a = a + 1.0
        return cur

    @property
    def beta_s(self) -> float:
        return 1.0 if self.name == "M0" else float(self.coef[0])

    def as_dict(self) -> dict:
        if self.name == "M0":
            return {"model": "M0"}
        return {"model": self.name, "intercept": self.intercept, **dict(zip(self.names, map(float, self.coef)))}


def pairs(cfg: Config, t: str, tk: str, score: str = "score_obv_ar1", elig: str = "eligible_main",
          groups: tuple[str, ...] = OUTFIELD, pos_col: str = "position_group_season",
          same_group: bool = False) -> pd.DataFrame:
    tr = pd.read_parquet(cfg.out("transitions.parquet"))
    d = tr[(tr.t == t) & (tr.tk == tk) & tr[f"{elig}_t"] & tr[f"{elig}_tk"]].copy()
    if pos_col == "position_group_canonical":
        d["pos"] = d.position_group_canonical_t.fillna(d.position_group_season_t)
        d["pos_tk"] = d.position_group_canonical_tk.fillna(d.position_group_season_tk)
    else:
        d["pos"], d["pos_tk"] = d.position_group_season_t, d.position_group_season_tk
    d = d[d.pos.isin(groups)]
    if same_group:
        d = d[d.pos == d.pos_tk]
    d["s_t"], d["s_tk"], d["age_t"] = d[f"{score}_t"], d[f"{score}_tk"], d.age_jan1_t
    return d.dropna(subset=["s_t", "s_tk", "age_t"]).reset_index(drop=True)


def origin_pool(cfg: Config, t: str, score: str = "score_obv_ar1", elig: str = "eligible_main",
                groups: tuple[str, ...] = OUTFIELD, pos_col: str = "position_group_season") -> pd.DataFrame:
    """Todos los elegibles en t (estén o no en t+k): población a la que se le predice."""
    ps = pd.read_parquet(cfg.out("players_season_full.parquet"))
    d = ps[(ps.season == t) & ps[elig]].copy()
    d["pos"] = d.position_group_canonical.fillna(d.position_group_season) if pos_col == "position_group_canonical" \
        else d.position_group_season
    d = d[d.pos.isin(groups)]
    d["s_t"], d["age_t"] = d[score], d.age_jan1
    return d.dropna(subset=["s_t", "age_t"]).reset_index(drop=True)


def fit_all(train: pd.DataFrame, groups: tuple[str, ...], center: float) -> dict[str, Fitted]:
    return {m: Fitted(m, groups, center).fit(train.s_t, train.age_t, train.pos, train.s_tk) for m in MODELS}


def oof_residuals(train: pd.DataFrame, model: str, groups: tuple[str, ...], center: float, seed: int) -> np.ndarray:
    res = np.zeros(len(train))
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(train):
        f = Fitted(model, groups, center).fit(train.s_t.values[tr], train.age_t.values[tr], train.pos.values[tr],
                                             train.s_tk.values[tr])
        res[te] = train.s_tk.values[te] - f.predict(train.s_t.values[te], train.age_t.values[te], train.pos.values[te])
    return res


def run(cfg: Config) -> dict:
    c = float(cfg.raw["models"]["age_center"])
    a, b = cfg.raw["design"]["training_transition"]
    train = pairs(cfg, a, b)
    fits = fit_all(train, OUTFIELD, c)
    out = {"training": f"{a}->{b}", "n_train": int(len(train)),
           "n_train_by_pos": train.pos.value_counts().to_dict(),
           "fits": {m: f.as_dict() for m, f in fits.items()}}
    # coeficientes de M2 con IC bootstrap de pares (P7)
    rng = np.random.default_rng(cfg.seed)
    B = int(cfg.raw["metrics"]["bootstrap_B"])
    boot = []
    for _ in range(B):
        ii = rng.integers(0, len(train), len(train))
        f = Fitted("M2", OUTFIELD, c).fit(train.s_t.values[ii], train.age_t.values[ii], train.pos.values[ii],
                                         train.s_tk.values[ii])
        boot.append(f.coef)
    boot = np.array(boot)
    out["M2_coef_ci95"] = {n: [float(np.quantile(boot[:, i], 0.025)), float(np.quantile(boot[:, i], 0.975))]
                           for i, n in enumerate(fits["M2"].names)}
    ages = np.arange(18, 36.01, 0.5)
    curve = np.array([bb[-2] * (ages - c) + bb[-1] * (ages - c) ** 2 for bb in boot])
    out["M2_age_curve"] = {"ages": ages.tolist(),
                           "point": (fits["M2"].coef[-2] * (ages - c) + fits["M2"].coef[-1] * (ages - c) ** 2).tolist(),
                           "lo": np.quantile(curve, 0.025, axis=0).tolist(), "hi": np.quantile(curve, 0.975, axis=0).tolist()}
    # predicciones (población de origen completa; la evaluación filtra luego)
    preds = []
    for k in range(1, int(cfg.raw["design"]["max_k"]) + 1):
        for t, tk in cfg.transitions(k):
            if (t, tk) == (a, b):
                continue
            pool = origin_pool(cfg, t)
            for m, f in fits.items():
                preds.append(pd.DataFrame({"player_id": pool.player_id, "model": m, "origin": t, "target": tk, "k": k,
                                           "pos": pool.pos, "s_t": pool.s_t, "age_t": pool.age_t,
                                           "pred": f.project(pool.s_t, pool.age_t, pool.pos, k)}))
    pd.concat(preds, ignore_index=True).to_parquet(cfg.out("predictions.parquet"), index=False)
    cfg.out("model_fits.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    r = run(load_config())
    print(json.dumps({k: r[k] for k in ("training", "n_train", "n_train_by_pos", "fits", "M2_coef_ci95")}, indent=1))

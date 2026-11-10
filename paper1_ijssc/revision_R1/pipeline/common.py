"""Utilidades compartidas del pipeline R1 (SPEC_R1 §5).

Lee la configuración, resuelve rutas y carga en modo solo lectura los módulos del pipeline enviado (P01, P03)
para reutilizar sus constantes. Ningún módulo de este paquete escribe fuera de `outputs/r1`, `results/` o la caché.
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import yaml

REPO = Path(__file__).resolve().parents[3]
R1 = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = R1 / "config" / "r1.yaml"
OUTFIELD = ("DEF", "MID", "FWD")
POSITIONS = ("GK", "DEF", "MID", "FWD")


@dataclass(frozen=True, slots=True)
class Season:
    label: str
    season_id: int
    name: str
    jan1: str


@dataclass(frozen=True)
class Config:
    raw: dict
    seasons: tuple[Season, ...]
    paths: dict[str, Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        labels = [s.label for s in self.seasons]
        if len(set(labels)) != len(labels):
            raise ValueError(f"etiquetas de temporada duplicadas: {labels}")
        tr = self.raw["design"]["training_transition"]
        if tr != labels[:2]:
            raise ValueError("la transición de entrenamiento debe ser las dos primeras temporadas (SPEC §5.4)")

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    def season(self, label: str) -> Season:
        for s in self.seasons:
            if s.label == label:
                return s
        raise KeyError(label)

    @property
    def labels(self) -> list[str]:
        return [s.label for s in self.seasons]

    def transitions(self, k: int) -> list[tuple[str, str]]:
        """Pares (t, t+k) disponibles en orden cronológico."""
        lab = self.labels
        return [(lab[i], lab[i + k]) for i in range(len(lab) - k)]

    def out(self, name: str) -> Path:
        p = self.paths["outputs"] / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def res(self, name: str) -> Path:
        p = self.paths["results"] / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def cache(self, name: str) -> Path:
        p = self.paths["cache"] / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


def _resolve(p: str) -> Path:
    return Path(p).expanduser() if p.startswith("~") else (REPO / p)


def load_config(path: Path | str | None = None, overrides: dict | None = None) -> Config:
    raw = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
    if overrides:
        for k, v in overrides.items():
            node = raw
            keys = k.split(".")
            for kk in keys[:-1]:
                node = node[kk]
            node[keys[-1]] = v
    seasons = tuple(Season(**s) for s in raw["seasons"])
    paths = {k: _resolve(v) for k, v in raw["paths"].items()}
    return Config(raw=raw, seasons=seasons, paths=paths)


def _load_module(name: str, path: Path) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@lru_cache(maxsize=None)
def p03() -> ModuleType:
    """Módulo P03 enviado, solo para leer constantes y funciones (no se modifica)."""
    return _load_module("P03_submitted", REPO / "scripts/pipeline/P03_score_computation.py")


@lru_cache(maxsize=None)
def p01() -> ModuleType:
    return _load_module("P01_submitted", REPO / "scripts/pipeline/P01_canonical_positions.py")


def pos_group(detail: str | None) -> str | None:
    """Posición granular de StatsBomb -> grupo (mismo mapeo que P01, con respaldo por palabras clave)."""
    if not detail:
        return None
    pm = p01().POS_MAP
    if detail in pm:
        return pm[detail]
    if "Goalkeeper" in detail:
        return "GK"
    if "Midfield" in detail:
        return "MID"
    if "Back" in detail or "Sweeper" in detail:
        return "DEF"
    if any(w in detail for w in ("Wing", "Forward", "Striker")):
        return "FWD"
    return None

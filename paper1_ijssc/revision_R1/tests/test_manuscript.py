"""Pruebas de la Fase 3 (T3.4) sobre el manuscrito revisado.

- terminología (SPEC §9.2, REQ-W2);
- anti-literales: ninguna cifra de resultados escrita a mano en las secciones (REQ-R4);
- extensión y formato: cuerpo <= 6.000 palabras sin flotantes, abstract de un párrafo, >= 5 keywords (REQ-W5);
- macros \\R... usadas y definidas;
- respuesta y carta sin etiquetas [PLANNED]/[DECISION] ni marcadores pendientes;
- ortografía británica, guiones largos, palabras duplicadas y destinos anteriores, con las reglas de
  paper1_ijssc/check_language.py (se importa, no se copia).
"""
from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
from pathlib import Path

import pytest

R1 = Path(__file__).resolve().parents[1]
MS = R1 / "manuscript"
SECTIONS = sorted((MS / "sections").glob("0*.tex"))
BODY = [p for p in SECTIONS if re.match(r"0[1-5]_", p.name)]  # Introduction a Conclusions
ABSTRACT = MS / "sections" / "00_abstract.tex"
ALL_TEX = [MS / "main.tex", MS / "supplementary.tex", MS / "response_to_reviewers.tex", MS / "cover_letter.tex",
           *SECTIONS]
WORD_LIMIT = 6000


def _check_language():
    spec = importlib.util.spec_from_file_location("check_language", R1.parent / "check_language.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _no_comments(s: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", s)


def _no_comment_env(s: str) -> str:
    # en la respuesta, los comentarios de los revisores se citan en el entorno comment
    return re.sub(r"\\begin\{comment\}.*?\\end\{comment\}", " ", s, flags=re.S)


def _prose(s: str) -> str:
    """Como check_language.prose(), sobre un texto ya leído; cada cifra queda como NUM."""
    s = re.sub(r"^\s*%.*$", "", s, flags=re.M)
    s = re.sub(r"\$[^$]*\$", " NUM ", s)
    s = re.sub(r"\\[RN][A-Z][A-Za-z]+(\{\})?", " NUM ", s)  # una cifra por macro: evita falsos 'and and'
    for _ in range(3):
        s = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?(\{[^{}]*\})?", " ", s)
    return s.replace("{", " ").replace("}", " ").replace("&", " ")


# --- terminología (§9.2) --------------------------------------------------------------------------------------------
TERMS = [
    (r"event[- ]level tracking|event[- ]tracking data", "usar 'event data'"),
    (r"\b360\b", "under_pressure es un atributo del evento (Data Specification v1.1), no de 360"),
    (r"notational and video-based", "usar 'notational analysis (usually video-based)'"),
    (r"broadcast- and optical-tracking", "describir positional (tracking) data por su fuente"),
    (r"orthogonal|distinct (axis|axes|dimension)", "usar 'weakly correlated' con su rho"),
    (r"retest reliability", "usar 'year-on-year persistence'"),
    (r"ranking horizon", "usar 'model-implied rank consistency'"),
    (r"\bfidelity\b|\bprecise ranking|regimes? of use|not justified by (the )?evidence", "reportar métricas con IC"),
    (r"peak age|age of peak", "usar 'age effect' con IC"),
    (r"\bEPL\b", "usar 'Premier League'"),
    (r"\bPSR\b|pressure success rate", "PSR solo en el apéndice del suplemento"),
]


@pytest.mark.parametrize("path", SECTIONS, ids=lambda p: p.name)
def test_terminology(path):
    text = _no_comments(path.read_text(encoding="utf-8"))
    hits = [f"{pat!r} -> {why}" for pat, why in TERMS if re.search(pat, text, flags=re.I)]
    assert not hits, f"{path.name}: " + "; ".join(hits)


# --- anti-literales (REQ-R4) ----------------------------------------------------------------------------------------
# Lista blanca: años y temporadas; convenciones que no son resultados (IC del 95 %, por 90 minutos, escala de 0 a 100,
# edad al 1 de enero); nombres de modelos y fórmulas (M0-M3, AR(1), k = 1, 2); apartados del suplemento (S1-S8) y el
# número de proyecto del financiamiento. Toda otra cifra debe venir de results/numbers.tex o results/design_numbers.tex.
LITERAL = re.compile(r"(?<![A-Za-z\\/])\d+(?:[.,/]\d+)*")
ALLOW = [re.compile(r"(19|20)\d\d(/\d\d)?")]
ALLOW_CONTEXT = re.compile(r"\bM[0-3]\b|AR\(1\)|k\s*=\s*[12]\b|\bS[1-8]\b|\b95\s*\\%|\bper\s+90\b|\bby\s+90\b"
                           r"|\b0 to 100\b|\b1 January\b|grant number \d+")


def _literals(path: Path) -> list[str]:
    s = _no_comments(path.read_text(encoding="utf-8"))
    s = re.sub(r"\\(autocite|cite|label|ref|input|includegraphics|url|href|eqref|cref|Cref)\*?(\[[^\]]*\])?\{[^}]*\}",
               " ", s)
    s = re.sub(r"\\R[A-Za-z]+(\{\})?", " ", s)  # macros del registro de cifras
    s = re.sub(r"\\[A-Za-z]+", " ", s)
    # la lista blanca se aplica al texto completo (una frase puede cortarse entre líneas), conservando los saltos
    s = ALLOW_CONTEXT.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), s)
    out = []
    for i, line in enumerate(s.splitlines(), start=1):
        clean = line
        for m in LITERAL.finditer(clean):
            tok = m.group(0)
            if any(a.fullmatch(tok) for a in ALLOW):
                continue
            out.append(f"{path.name}:{i}: {tok!r} en ...{clean[max(0, m.start() - 30):m.end() + 15].strip()}...")
    return out


@pytest.mark.parametrize("path", SECTIONS, ids=lambda p: p.name)
def test_no_result_literals(path):
    hits = _literals(path)
    assert not hits, "\n".join(hits)


# --- macros usadas y definidas --------------------------------------------------------------------------------------
def _defined_macros() -> set[str]:
    out = set()
    for f in (R1 / "results" / "numbers.tex", R1 / "results" / "design_numbers.tex"):
        out |= set(re.findall(r"\\newcommand\{\\(R[A-Za-z]+|N[A-Za-z]+)\}", f.read_text(encoding="utf-8")))
    return out


def test_macros_defined():
    defined = _defined_macros()
    used = set()
    for p in ALL_TEX:
        used |= set(re.findall(r"\\(R[A-Z][A-Za-z]+)", _no_comments(p.read_text(encoding="utf-8"))))
    missing = sorted(m for m in used if m not in defined and not m.startswith(("Require", "RequirePackage")))
    assert not missing, f"macros sin definir: {missing}"


# --- extensión y formato (REQ-W5) -----------------------------------------------------------------------------------
def body_word_count() -> int:
    joined = "\n".join(_no_comments(p.read_text(encoding="utf-8")) for p in BODY)
    joined = re.sub(r"\\begin\{(table|figure)\*?\}.*?\\end\{\1\*?\}", " ", joined, flags=re.S)
    if shutil.which("detex"):
        r = subprocess.run(["detex"], input=joined, capture_output=True, text=True, check=True)
        return len(r.stdout.split())
    return len(_prose(joined).split())


def test_word_count():
    n = body_word_count()
    assert n <= WORD_LIMIT, f"cuerpo: {n} palabras (> {WORD_LIMIT})"


def test_abstract_single_paragraph_and_keywords():
    s = _no_comments(ABSTRACT.read_text(encoding="utf-8"))
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", s, flags=re.S)
    assert m, "no hay entorno abstract"
    body = re.sub(r"\\(begin|end)\{revised\}", "", m.group(1)).strip()
    assert not re.search(r"\n\s*\n", body), "el abstract tiene más de un párrafo"
    src = s + _no_comments((MS / "main.tex").read_text(encoding="utf-8"))
    k = re.search(r"\\keywords\{([^}]*)\}|Keywords?[:}]\s*([^\n]*)", src, flags=re.I)
    assert k, "no hay keywords"
    kws = [w for w in re.split(r"[;,]", k.group(1) or k.group(2)) if w.strip()]
    assert len(kws) >= 5, f"keywords: {len(kws)}"


# --- respuesta y carta ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["response_to_reviewers.tex", "cover_letter.tex", "supplementary.tex"])
def test_no_pending_tags(name):
    s = _no_comments((MS / name).read_text(encoding="utf-8"))
    hits = re.findall(r"\[(PLANNED|DECISION|DONE[^\]]*)\]|\bTODO\b|\bTBD\b|\bFIXME\b|\bXXX\b|\?\?", s)
    assert not hits, f"{name}: {hits}"


# --- reglas de lenguaje de check_language.py -----------------------------------------------------------------------
@pytest.mark.parametrize("path", ALL_TEX, ids=lambda p: p.name)
def test_language(path):
    cl = _check_language()
    raw = path.read_text(encoding="utf-8")
    if path.name == "response_to_reviewers.tex":  # las citas de los revisores no son texto nuestro
        raw = _no_comment_env(raw)
    text = _prose(raw)
    low = text.lower()
    hits = []
    for m in re.finditer(r"\b([a-z]{3,})\s+\1\b", text):
        hits.append(f"palabra duplicada '{m.group(0)}'")
    for br, am in cl.BRITISH.items():
        for m in re.finditer(rf"\b\w*{am}\w*\b", low):
            if m.group(0) not in cl.SPELLING_ALLOW:
                hits.append(f"'{m.group(0)}' americano; usar '{br}...'")
                break
    if re.search(r"\btoward\b", low):
        hits.append("'toward' sin s")
    if "\u2014" in text or re.search(r"(?<!-)---(?!-)", text):
        hits.append("guion largo")
    for i, line in enumerate(_no_comments(raw).splitlines(), start=1):
        if cl.STALE_VENUE.search(line) and not cl.VENUE_ALLOW.search(line):
            hits.append(f"línea {i}: destino anterior -> {line.strip()[:60]}")
    assert not hits, f"{path.name}: " + "; ".join(hits)

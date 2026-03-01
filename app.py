""" tester łaciny - aplikacja do wyszukiwania potencjalnych błędów w tekście odczytanym przez narzędzia OCR / HTR """
import os
import re
import json
import html
import time
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from itertools import islice
from dotenv import load_dotenv
from flask import Flask, render_template, request
from markupsafe import Markup
from openai import OpenAI


app = Flask(__name__)

# wczytanie API-KEY i innych zmiennych środowiskowych
env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)
OPENAI_ORG_ID = os.environ.get('OPENAI_ORG_ID')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

DEFAULT_ENGINE = os.getenv("DEFAULT_ENGINE", "local")  # local | llm
DEFAULT_MODEL_LLM = os.getenv("OPENAI_MODEL", "gpt-5-mini")
MAX_ISSUES = 80

LOCAL_SUGGEST_LIMIT = 120        # maksymalnie dla tylu tokenów licz sugestie
LOCAL_MAX_TOKEN_FOR_SUGGEST = 25 # nie licz sugestii dla bardzo długich tokenów
LOCAL_SUGGEST_K = 3              # ile sugestii pobierać

_HUNSPELL_DICT = None
_LATIN_LEMMATIZER = None

# ścieżka bazowa do słownika Hunspell (bez .aff/.dic)
HUNSPELL_BASE = os.getenv("HUNSPELL_BASE", os.path.join("dicts", "la", "la"))

WORD_RE = re.compile(r"[A-Za-z\u00C0-\u024F\u1E00-\u1EFF\u017F]+")

MAX_CHARS = 5000 # maksymalna liczba znaków w polu tekstowym do przetwarzania

client = OpenAI(api_key=OPENAI_API_KEY)

# schemat dla trybu LLM
OCR_ISSUES_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "issues": {
            "type": "array",
            "maxItems": MAX_ISSUES,
            "items": {
                "type": "object",
                "properties": {
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                    "category": {
                        "type": "string",
                        "enum": [
                            "glyph_confusion",
                            "segmentation",
                            "abbreviation",
                            "lexical",
                            "syntax",
                            "non_latin_char",
                            "other",
                        ],
                    },
                    "excerpt": {"type": "string"},
                    "suggestion": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": [
                    "severity",
                    "category",
                    "excerpt",
                    "suggestion",
                    "reason",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["issues"],
    "additionalProperties": False,
}

CONFUSION_MAP = [
    ("ſ", "s"), ("f", "ſ"), ("u", "v"), ("v", "u"),
    ("i", "j"), ("j", "i"), ("m", "in"), ("in", "m"),
    ("rn", "m"), ("m", "rn"), ("vv", "w"), ("cl", "d"),
    ("ae", "æ"), ("oe", "œ")
]


# -------------------------------- FUNCTIONS ----------------------------------
def get_hunspell_dict():
    """ inicjalizacja słownika Hunspell."""
    global _HUNSPELL_DICT
    if _HUNSPELL_DICT is None:
        from spylls.hunspell import Dictionary
        # Ładowanie z dysku nastąpi tylko raz
        _HUNSPELL_DICT = Dictionary.from_files(HUNSPELL_BASE)
    return _HUNSPELL_DICT

def get_latin_lemmatizer():
    """ inicjalizacja lematyzatora CLTK """
    global _LATIN_LEMMATIZER
    if _LATIN_LEMMATIZER is None:
        from cltk.lemmatize.lat import LatinBackoffLemmatizer
        _LATIN_LEMMATIZER = LatinBackoffLemmatizer()
    return _LATIN_LEMMATIZER

def check_confusion_variants(token: str) -> Optional[str]:
    """próbuje podmienić znaki według matrycy i sprawdza w słowniku."""
    D = get_hunspell_dict()
    for source, target in CONFUSION_MAP:
        if source in token.lower():
            variant = token.lower().replace(source, target)
            if D.lookup(variant):
                return variant
    return None

def hunspell_suggest_cached(base: str, max_n: int) -> Tuple[str, ...]:
    """ przygotowaie sugestii poprawnych słów """
    D = get_hunspell_dict()
    return tuple(islice(D.suggest(base), max_n))

def lemmatize_token(token_lower: str) -> Optional[str]:
    """ lematyzacja """
    try:
        lem = get_latin_lemmatizer()
        res = lem.lemmatize([token_lower])
        return res[0][1] if res else None
    except Exception:
        return None
    
def edit_distance(a: str, b: str, limit: int = 3) -> int:
    """Levenshtein """
    if a == b:
        return 0
    if abs(len(a) - len(b)) >= limit:
        return limit
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(a) + 1))
    for i, ch_b in enumerate(b, start=1):
        curr = [i]
        min_row = curr[0]
        for j, ch_a in enumerate(a, start=1):
            cost = 0 if ch_a == ch_b else 1
            curr.append(
                min(
                    prev[j] + 1,       # del
                    curr[j - 1] + 1,   # ins
                    prev[j - 1] + cost # sub
                )
            )
            if curr[j] < min_row:
                min_row = curr[j]
        if min_row >= limit:
            return limit
        prev = curr
    return prev[-1]

@lru_cache(maxsize=50000)
def hunspell_lookup_cached(token: str) -> Tuple[bool, str]:
    return hunspell_lookup(token)

def normalize_variants(token: str) -> List[str]:
    """
    generuje warianty normalizacyjne (dla sprawdzania), aby zmniejszać fałszywe alarmy przy u/v, i/j, ſ/s, ligaturach.
    """
    t0 = token
    t = token.lower()

    variants = {t0, t}

    # długie s
    variants.add(t.replace("ſ", "s"))

    # ligatury
    variants.add(t.replace("æ", "ae").replace("œ", "oe"))
    variants.add(t.replace("Æ", "Ae").replace("Œ", "Oe"))

    # i/j
    variants.add(t.replace("j", "i"))
    variants.add(t.replace("i", "j"))

    # u/v (w obie strony)
    variants.add(t.replace("v", "u"))
    variants.add(t.replace("u", "v"))

    # kombinacje częste
    variants.add(t.replace("j", "i").replace("v", "u"))
    variants.add(t.replace("j", "i").replace("u", "v"))
    variants.add(t.replace("ſ", "s").replace("v", "u"))

    # usuń puste i duplikaty
    out = [v for v in variants if v]
    # lekka preferencja: najpierw lower
    out.sort(key=lambda x: (x != t, len(x)))
    return out


def hunspell_lookup(token: str) -> Tuple[bool, str]:
    """
    sprawdza token w Hunspell (przez warianty normalizacyjne).
    zwraca: (czy_ok, wariant_który_przeszedł_lub_pusty)
    """
    D = get_hunspell_dict()
    for v in normalize_variants(token):
        if D.lookup(v):
            return True, v
    return False, ""


# CLTK (lematyzacja)
@lru_cache(maxsize=1)
def load_latin_lemmatizer():
    """
    CLTK v1+: LatinBackoffLemmatizer
    """
    from cltk.lemmatize.lat import LatinBackoffLemmatizer
    return LatinBackoffLemmatizer()


def hunspell_suggest(token: str, max_n: int = 5) -> List[str]:
    #  rezygnancja jeżeli długie słowo - za długie obliczenia
    if len(token) > 10:
        return []
    
    base = normalize_variants(token)[0]
    return list(hunspell_suggest_cached(base, max_n))


def is_probable_abbrev(text: str, start: int, end: int, token: str) -> bool:
    """
    Heurystyka: skrót w rodzaju 'V.' / 'D.' / 'Des.' itd.
    Uznajemy za skrót, jeśli:
      - token jest krótki (<=3) i zaczyna się wielką literą
      - bezpośrednio po tokenie stoi kropka
    """
    if len(token) <= 3 and token[:1].isupper():
        return end < len(text) and text[end] == "."
    return False


def downgrade(sev: str) -> str:
    return {"high": "medium", "medium": "low", "low": "low"}.get(sev, "low")


def local_analyze(text: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Zwraca: (issues, warnings)
    issues: lista obiektów zgodnych z formatem podświetlania
    warnings: komunikaty, np. o braku CLTK lub słownika
    """
    warnings: List[str] = []

    # sprawdź dostępność Hunspell
    try:
        _ = get_hunspell_dict()
    except Exception as e:
        warnings.append(
            f"Nie udało się załadować słownika Hunspell z bazy: {HUNSPELL_BASE}. "
            f"Upewnij się, że istnieją pliki {HUNSPELL_BASE}.aff i {HUNSPELL_BASE}.dic. "
            f"Szczegóły: {e}"
        )
        return [], warnings

    # CLTK jest „wspomagające”: jeśli nie działa będzie użyty sam Hunspell
    cltk_ok = True
    try:
        _ = load_latin_lemmatizer()
    except Exception as e:
        cltk_ok = False
        warnings.append(
            "CLTK nie jest dostępny albo nie ma wymaganych zasobów; analiza lokalna "
            "będzie wykonywana bez lematyzacji. Szczegóły: "
            f"{e}"
        )

    issues: List[Dict[str, Any]] = []
    tokens: List[Tuple[int, int, str, bool]] = []  # (start,end,token,ok)

    suggest_count = 0
    print('Początek analizy....')

    # detekcja znaków „śmieciowych” (częste w OCR): U+FFFD itp.
    for i, ch in enumerate(text):
        if ch == "\uFFFD":
            issues.append(
                {
                    "start": i,
                    "end": i + 1,
                    "severity": "high",
                    "category": "other",
                    "excerpt": text[i : i + 1],
                    "suggestion": "",
                    "reason": "Znak zastępczy U+FFFD (często efekt błędu kodowania/OCR).",
                }
            )

    # tokenizacja + hunspell
    for m in WORD_RE.finditer(text):
        print(m)
        start, end = m.start(), m.end()
        token = m.group(0)
        token_lower = token.lower()

        # krótki token (np. 'V') rozpatrywany tylko jako skrót
        abbrev = is_probable_abbrev(text, start, end, token)

        ok, ok_variant = (True, token_lower) if abbrev else hunspell_lookup_cached(token)

        tokens.append((start, end, token, ok))

        if ok:
            continue

        # jeśli wygląda na skrót, oznaczane informacyjnie, a nie jako błąd
        if abbrev:
            issues.append(
                {
                    "start": start,
                    "end": end + 1 if end < len(text) and text[end] == "." else end,
                    "severity": "low",
                    "category": "abbreviation",
                    "excerpt": text[start : (end + 1 if end < len(text) and text[end] == "." else end)],
                    "suggestion": "",
                    "reason": "Prawdopodobny skrót (litera/krótki segment z kropką).",
                }
            )
            continue

        # zamiany znaków
        confusion_fix = check_confusion_variants(token)
        if confusion_fix:
            issues.append({
                "start": start,
                "end": end,
                "excerpt": token,
                "severity": "high",
                "category": "glyph_confusion",
                "suggestion": confusion_fix,
                "reason": f"Prawdopodobna pomyłka znaków (typowa dla OCR/HTR)."
            })
            continue
        
        suggestion = ""
        reason_parts: List[str] = []

        lemma = lemmatize_token(token_lower) if cltk_ok else None

        if lemma:
            # jeśli lemma jest rozpoznawalna słownikowo, zmniejszenie podejrzenie
            lemma_ok, _ = hunspell_lookup_cached(lemma)
            
            if lemma_ok:
                reason_parts.append(f"Lematyzacja sugeruje lemma: {lemma}.")
            else:
                reason_parts.append(f"Lematyzacja sugeruje lemma: {lemma} (nieznane słownikowo).")

        # w pętli tokenów:
        suggs = []
        if (suggest_count < LOCAL_SUGGEST_LIMIT) and (len(token) <= LOCAL_MAX_TOKEN_FOR_SUGGEST):
            suggs = hunspell_suggest(token, max_n=LOCAL_SUGGEST_K)
            if suggs:
                suggest_count += 1

        if suggs:
            print(f'Znaleziono sugestie: {len(suggs)}')
            suggestion = suggs[0]
            d = edit_distance(token_lower, suggestion.lower(), limit=3)
            reason_parts.append(f"Nie znaleziono w słowniku; sugestia: {suggestion} (odległość≈{d}).")
            if d <= 1:
                severity = "high"
                category = "glyph_confusion"
            elif d == 2:
                severity = "medium"
                category = "glyph_confusion"
            else:
                severity = "low"
                category = "lexical"
        else:
            # próba rozcinania słów
            split_suggestion = None
            for k in range(2, len(token_lower) - 2):
                left, right = token_lower[:k], token_lower[k:]

                left_ok, _ = hunspell_lookup_cached(left)
                right_ok, _ = hunspell_lookup_cached(right)
                if left_ok and right_ok:
                    split_suggestion = f"{token[:k]} {token[k:]}"
                    break

            if split_suggestion:
                suggestion = split_suggestion
                severity = "medium"
                category = "segmentation"
                reason_parts.append("Możliwy błąd segmentacji: rozcięcie daje dwie formy słownikowe.")
            else:
                severity = "medium"
                category = "lexical"
                reason_parts.append("Nie znaleziono w słowniku i brak sugestii Hunspell.")

        # jeśli lemma wygląda sensownie, obniżenie wagi błędu o jeden poziom
        if lemma and ("lemma" in " ".join(reason_parts).lower()):
            severity = downgrade(severity)

        issues.append(
            {
                "start": start,
                "end": end,
                "severity": severity,
                "category": category,
                "excerpt": text[start:end],
                "suggestion": suggestion,
                "reason": " ".join(reason_parts).strip(),
            }
        )

    # „zlane dwa wyrazy”: jeśli token_i + token_{i+1} jest słownikowy
    # (i oba osobno nie są)
    spans_seen = {(it["start"], it["end"]) for it in issues}
    for i in range(len(tokens) - 1):
        s1, e1, t1, ok1 = tokens[i]
        s2, e2, t2, ok2 = tokens[i + 1]

        if ok1 or ok2:
            continue

        between = text[e1:s2]
        joined = (t1 + t2).lower()

        joined_ok, _ = hunspell_lookup_cached(joined)
        if joined_ok:
            span = (s1, e2)
            if span in spans_seen:
                continue
            spans_seen.add(span)
            issues.append(
                {
                    "start": s1,
                    "end": e2,
                    "severity": "medium",
                    "category": "segmentation",
                    "excerpt": text[s1:e2],
                    "suggestion": t1 + ("" if between == "" else between) + t2,
                    "reason": "Możliwy błąd segmentacji: połączenie z następnym segmentem daje formę słownikową.",
                }
            )

    issues.sort(key=lambda x: (x["start"], x["end"]))
    return issues[:MAX_ISSUES], warnings


# ścieżka analizy przez LLM
def call_llm_for_issues(text: str, model: str) -> List[Dict[str, Any]]:
    system_msg = (
        "Jesteś specjalistą od łacińskich tekstów rękopiśmiennych i druków z XV/XVI wieku "
        "oraz typowych błędów OCR/HTR. Twoim celem jest WYŁĄCZNIE wskazanie fragmentów "
        "prawdopodobnie błędnych (lub wysoce podejrzanych) w transkrypcji OCR/HTR.\n\n"
        "Zasady:\n"
        "1) Zachowaj ostrożność: warianty ortograficzne (u/v, i/j, ligatury) nie są automatycznie błędem.\n"
        "2) Jeśli oznaczasz fragment, podaj start/end jako indeksy znaków w tekście wejściowym.\n"
        "3) 'excerpt' MUSI być dokładnie równy text[start:end].\n"
        "4) Pole 'suggestion' ZAWSZE zwracaj: jeśli brak propozycji, ustaw \"\".\n"
        "5) Preferuj krótkie fragmenty (1-20 znaków), chyba że kontekst wymaga więcej.\n"
        "6) Nie poprawiaj całego tekstu; zwróć tylko listę podejrzanych miejsc.\n"
    )

    user_msg = (
        "Przeanalizuj poniższy tekst OCR/HTR i zwróć listę podejrzanych fragmentów.\n\n"
        "TEKST (nie zmieniaj go, indeksuj dokładnie po znakach):\n"
        "```text\n"
        f"{text}\n"
        "```"
    )

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        reasoning={"effort": "low"},
        text={
            "format": {
                "type": "json_schema",
                "name": "ocr_issues",
                "schema": OCR_ISSUES_SCHEMA,
                "strict": True,
            }
        },
    )

    data = json.loads(resp.output_text)
    return data.get("issues", [])


def sanitize_and_sort_issues(text: str, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    clean: List[Dict[str, Any]] = []
    # kursor do przeszukiwania tekstu, aby uniknąć łapania ciągle tego samego słowa
    search_cursor = 0

    for it in issues:
        excerpt = it.get("excerpt", "").strip()
        if not excerpt:
            continue
            
        # szukanie fragmentu w tekście, zaczynając od ostatniego znalezionego miejsca
        found_pos = text.find(excerpt, search_cursor)
        
        # Jeśli nie znaleziono przed, próba od początku (na wypadek pomyłki LLM)
        if found_pos == -1:
            found_pos = text.find(excerpt, 0)

        if found_pos != -1:
            it["start"] = found_pos
            it["end"] = found_pos + len(excerpt)
            
            # aktualizacja kursora
            search_cursor = it["end"]
            
            if "suggestion" not in it:
                it["suggestion"] = ""
            clean.append(it)

    # sortowanie ROSNĄCO (od początku tekstu do końca).
    # by funkcja highlight_html przeszła przez wszystkie fragmenty poprawnie
    clean.sort(key=lambda x: x["start"])
    return clean[:MAX_ISSUES]


def highlight_html(text: str, issues: List[Dict[str, Any]]) -> Markup:
    pieces: List[str] = []
    cursor = 0

    for idx, it in enumerate(issues):
        s, e = it["start"], it["end"]
        if s < cursor:
            continue

        pieces.append(html.escape(text[cursor:s]))

        sev = it.get("severity", "low")
        cat = it.get("category", "other")
        reason = it.get("reason", "")
        suggestion = it.get("suggestion", "")

        tooltip = f"{cat}: {reason}"
        if suggestion:
            tooltip += f" | sugestia: {suggestion}"
        tooltip = html.escape(tooltip[:400])

        marked = html.escape(text[s:e])
        pieces.append(f'<mark id="m{idx}" class="sev-{sev}" title="{tooltip}">{marked}</mark>')
        cursor = e

    pieces.append(html.escape(text[cursor:]))
    return Markup("".join(pieces))


# Routes
@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        engine=DEFAULT_ENGINE,
        model=DEFAULT_MODEL_LLM,
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    text = request.form.get("text", "")

    #  walidacja długości tekstu
    if len(text) > MAX_CHARS:
        return render_template(
            "index.html",
            engine=request.form.get("engine", "local"),
            model=DEFAULT_MODEL_LLM,
            text=text[:MAX_CHARS], # przytnij tekst do wyświetlenia
            error=f"Tekst jest za długi. Maksymalna dozwolona liczba znaków to {MAX_CHARS}."
        )

    engine = (request.form.get("engine", DEFAULT_ENGINE) or DEFAULT_ENGINE).strip().lower()

    if not text.strip():
        return render_template(
            "index.html",
            engine=engine,
            model=DEFAULT_MODEL_LLM,
            error="Brak tekstu wejściowego.",
        )

    try:
        warnings: List[str] = []
        if engine == "llm":
            raw_issues = call_llm_for_issues(text, model=DEFAULT_MODEL_LLM)
            issues = sanitize_and_sort_issues(text, raw_issues)
        else:
            issues, warnings = local_analyze(text)
            issues = sanitize_and_sort_issues(text, issues)

        highlighted = highlight_html(text, issues)

        return render_template(
            "index.html",
            engine=engine,
            model=DEFAULT_MODEL_LLM,
            text=text,
            issues=issues,
            highlighted=highlighted,
            warnings=warnings,
            error=None,
        )
    except Exception as e:
        return render_template(
            "index.html",
            engine=engine,
            model=DEFAULT_MODEL_LLM,
            text=text,
            error=f"Błąd analizy: {e}",
        )


# -------------------------------- MAIN ---------------------------------------
if __name__ == "__main__":
    app.run(debug=True)

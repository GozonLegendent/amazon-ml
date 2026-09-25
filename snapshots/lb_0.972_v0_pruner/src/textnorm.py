"""Text normalisation for business names and addresses.

Everything here is rule-based and country-agnostic: no lookup tables of real
businesses or places. Abbreviation maps only encode generic spelling
conventions (Rd/Road, Ltd/Limited, R./Rue, SARL/S.A.R.L., ...).

Two views are produced per record:
  * ``mtext``  - light clean-up used as transformer input (keeps native scripts,
                 folds Latin accents, lower-cases, drops NULL placeholders).
  * feature tokens - aggressively canonicalised ASCII tokens used for string
                 similarity features (native-script tokens are transliterated
                 with a dictionary learned from the training pairs, falling
                 back to unidecode).
"""
import re
import unicodedata

from unidecode import unidecode

PLACEHOLDERS = {"null", "none", "n/a", "na", "nan", "nil", "-", "--", "unknown"}

# Canonical short forms. Keys are lower-case ascii tokens after punctuation split.
NAME_CANON = {
    "corporation": "corp", "incorporated": "inc", "company": "co", "limited": "ltd",
    "private": "pvt", "pvtltd": "pvt ltd", "llc": "llc", "l": "l", "and": "&",
    "et": "&", "cie": "co", "compagnie": "co", "societe": "ste", "society": "soc",
    "association": "assoc", "associates": "assoc", "services": "svc", "service": "svc",
    "international": "intl", "technologies": "tech", "technology": "tech",
    "enterprises": "ent", "enterprise": "ent", "industries": "ind", "industry": "ind",
    "brothers": "bros", "center": "ctr", "centre": "ctr", "groupe": "group",
    "etablissements": "ets", "saint": "st", "sainte": "ste",
}
LEGAL = {
    "inc", "corp", "co", "ltd", "pvt", "llc", "llp", "lp", "plc", "pllc", "pc", "pa",
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "scop", "selarl", "gie", "ets",
    "group", "holdings", "holding", "partners", "&", "the", "of", "ms", "m",
}
HONORIFIC = {"dr", "smt", "sri", "shri", "shree", "mr", "mrs", "ms", "the", "m s", "messrs"}
NAME_STOP = {"the", "of", "de", "du", "la", "le", "les", "des", "d", "l", "&", "a", "an"}

ADDR_CANON = {
    # English street types
    "road": "rd", "street": "st", "avenue": "ave", "av": "ave", "avn": "ave",
    "boulevard": "blvd", "bd": "blvd", "boul": "blvd", "drive": "dr", "court": "ct",
    "lane": "ln", "place": "pl", "terrace": "ter", "trail": "trl", "circle": "cir",
    "highway": "hwy", "parkway": "pkwy", "square": "sq", "suite": "ste", "apartment": "apt",
    "building": "bldg", "floor": "flr", "north": "n", "south": "s", "east": "e",
    "west": "w", "mount": "mt", "fort": "ft", "saint": "st", "sainte": "ste",
    "township": "twp", "point": "pt", "heights": "hts", "junction": "jct",
    # French street types
    "rue": "r", "allee": "all", "alle": "all", "impasse": "imp", "chemin": "che",
    "route": "rte", "faubourg": "fbg", "quai": "qu", "residence": "res", "cite": "cite",
    "general": "gen", "marechal": "mal", "president": "pdt", "docteur": "dr",
    # Indian address words
    "nagar": "ngr", "colony": "col", "sector": "sec", "near": "nr", "opposite": "opp",
    "opp": "opp", "industrial": "ind", "area": "area", "village": "vill", "vil": "vill",
    "district": "dist", "dt": "dist", "post": "po", "marg": "marg", "cross": "crs",
    "main": "main", "block": "blk", "phase": "ph", "extension": "extn", "ext": "extn",
}
ADDR_NOISE = {"no", "door", "hno", "house", "unit", "apt", "ste", "pmb", "po", "box",
              "flat", "dno", "shop", "office", "plot", "bis", "ter", "b"}

_COMB = re.compile(r"[̀-ͯ]")
_SPLIT = re.compile(r"[^0-9a-zऀ-෿]+")
_DIGITS = re.compile(r"\d+")
_DOMAIN = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9\-]+)\.(?:com|net|org|in|co|fr|biz|info|us|co\.in)$")
_LONGNUM = re.compile(r"\b\d{5,}\b")
_JUNK_PAREN = re.compile(r"\((?:id|ref|no)[^)]*\)|#\s*\d{4,}|\bid\s*[:#]\s*\d+", re.I)


def fold(s):
    """NFKC, strip Latin combining accents (keeps Indic vowel signs), lower-case."""
    if not s:
        return ""
    s = s.replace("‌", "").replace("‍", "")
    s = unicodedata.normalize("NFKD", s)
    s = _COMB.sub("", s)
    return unicodedata.normalize("NFC", s).lower()


def is_latin_token(t):
    return all(ord(c) < 0x250 for c in t)


def has_indic(s):
    return any(0x0900 <= ord(c) <= 0x0DFF for c in s)


def clean_address_raw(a):
    """Drop NULL-ish placeholder components; keep the rest verbatim."""
    if a is None:
        return ""
    parts = [p.strip() for p in a.split(",")]
    parts = [p for p in parts if p and p.strip().lower() not in PLACEHOLDERS]
    return ", ".join(parts)


def model_text(name, addr):
    """Text fed to the transformer models."""
    n = re.sub(r"\s+", " ", fold(name or "")).strip()
    a = re.sub(r"\s+", " ", fold(clean_address_raw(addr))).strip()
    return f"{n} ; {a}" if a else f"{n} ;"


def _merge_single_letters(toks):
    """['s','a','r','l'] -> ['sarl']; ['l','l','c'] -> ['llc']."""
    out, run = [], []
    for t in toks:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if len(run) >= 2:
            out.append("".join(run))
        else:
            out.extend(run)
        run = []
        out.append(t)
    if len(run) >= 2:
        out.append("".join(run))
    else:
        out.extend(run)
    return out


def _translit(tok, tdict):
    if is_latin_token(tok):
        return tok
    t = tdict.get(tok) if tdict else None
    if t is None:
        t = re.sub(r"[^a-z0-9 ]", "", unidecode(tok).lower())
    return t


def name_tokens(name, tdict=None):
    """Canonical ASCII token list for a business name.

    Returns (tokens, is_domain) where tokens is the full canonical list.
    """
    s = fold(name or "")
    s = _JUNK_PAREN.sub(" ", s)
    s = _LONGNUM.sub(" ", s)
    is_domain = False
    m = _DOMAIN.match(s.strip())
    if m:
        is_domain = True
        s = m.group(1).replace("-", " ")
    s = s.replace("&", " and ").replace("'", "").replace("’", "")
    toks = [t for t in _SPLIT.split(s) if t]
    toks = [_translit(t, tdict) for t in toks]
    toks = " ".join(toks).split()
    toks = _merge_single_letters(toks)
    out = []
    for t in toks:
        t = NAME_CANON.get(t, t)
        out.extend(t.split())
    return out, is_domain


def name_core(toks):
    """Name tokens minus legal suffixes / honorifics / stop-words, de-duplicated in order."""
    seen, out = set(), []
    for t in toks:
        if t in LEGAL or t in HONORIFIC or t in NAME_STOP or t.isdigit():
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def addr_tokens(addr, tdict=None):
    """Canonical ASCII token list and normalised number list for an address."""
    s = fold(clean_address_raw(addr))
    nums = [n.lstrip("0") or "0" for n in _DIGITS.findall(s)]
    s = s.replace("'", " ")
    toks = [t for t in _SPLIT.split(s) if t]
    toks = [_translit(t, tdict) for t in toks]
    toks = " ".join(toks).split()
    out = []
    for t in toks:
        if t.isdigit():
            continue
        # split alnum like 'b3' / 'srno58p' into alpha part only (digits are in nums)
        t = re.sub(r"\d+", " ", t).strip()
        for u in t.split():
            u = ADDR_CANON.get(u, u)
            if u not in ADDR_NOISE:
                out.append(u)
    return out, nums


def script_of(s):
    """Coarse script id of a string: 0 latin/empty, 1 indic, 2 other."""
    if not s:
        return 0
    if has_indic(s):
        return 1
    if any(ord(c) >= 0x250 and c.isalpha() for c in s):
        return 2
    return 0

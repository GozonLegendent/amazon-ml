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
    "etablissements": "ets", "saint": "st", "sainte": "ste", "frs": "freres",
}
LEGAL = {
    "inc", "corp", "co", "ltd", "pvt", "llc", "llp", "lp", "plc", "pllc", "pc", "pa",
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "scop", "selarl", "gie", "ets",
    "group", "holdings", "holding", "partners", "&", "the", "of", "ms", "m", "ei",
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
    # more French street types / building words
    "ch": "che", "chem": "che", "cours": "crs", "q": "qu", "passage": "psg", "pass": "psg",
    "bld": "blvd", "bvd": "blvd", "bldv": "blvd", "chaussee": "chs", "chee": "chs", "promenade": "prom",
    "lotissement": "lot", "hameau": "ham", "esplanade": "esp", "fg": "fbg", "resid": "res",
    "app": "apt", "appt": "apt", "appartement": "apt", "etage": "flr", "etg": "flr",
    "bat": "bldg", "batiment": "bldg",
}
ADDR_NOISE = {"no", "door", "hno", "house", "unit", "apt", "ste", "pmb", "po", "box",
              "flat", "dno", "shop", "office", "plot", "bis", "ter", "quater", "b"}
# French street types, used to repair typos ("Anenue", "Impase") in the slot right after the house number
STREET_TYPES = ["rue", "avenue", "boulevard", "chemin", "allee", "impasse", "route", "place", "quai", "cours",
                "cour", "passage", "square", "residence", "cite", "faubourg", "chaussee", "promenade",
                "lotissement", "hameau", "sentier", "sente", "esplanade", "parvis", "villa", "plage", "cote"]

_COMB = re.compile(r"[̀-ͯ]")
_SPLIT = re.compile(r"[^0-9a-zऀ-෿]+")
_DIGITS = re.compile(r"\d+")
_DOMAIN = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9\-]+)\.(?:com|net|org|in|co|fr|biz|info|us|co\.in)$")
_LONGNUM = re.compile(r"\b\d{5,}\b")
_JUNK_PAREN = re.compile(r"\((?:id|ref|no)[^)]*\)|#\s*\d{4,}|\bid\s*[:#]\s*\d+", re.I)
# "X dba Y" / "X formerly Y" -> the business is Y
_ALIAS = re.compile(r"\s+(?:d\.?/?b\.?/?a\.?:?|doing business as|trading as|t/a|a/k/a|aka|f/k/a|fka|"
                    r"formerly known as|formerly|nee|née)\s+", re.I)
_NUMERO = re.compile(r"\bn\s*[°º]\s*|\bno\.?\s*(?=\d)|#\s*(?=\d)")        # "N°24", "No.24", "#24" -> "24"
_NUM_SUFFIX = re.compile(r"\b(\d+)\s*(?:bis|ter|quater|[bt])\b(?!\.)")         # "5 bis", "1 T" -> "5", "1"


def fold(s):
    """NFKC, strip Latin combining accents (keeps Indic vowel signs), lower-case."""
    if not s:
        return ""
    s = s.replace("‌", "").replace("‍", "")
    s = unicodedata.normalize("NFKD", s)
    s = _COMB.sub("", s)
    return unicodedata.normalize("NFC", s).lower()


def _lig(s):
    return (s or "").replace("œ", "oe").replace("Œ", "Oe").replace("æ", "ae").replace("Æ", "Ae").replace("\x92", "'")


def name_main(name):
    """Right-hand side of an alias marker ('X dba Y' -> 'Y'), else the name itself."""
    p = _ALIAS.split(name or "", maxsplit=1)
    return p[1] if len(p) == 2 and p[1].strip() else (name or "")


def fuzzy_street_type(t):
    """Repair a misspelt street type (edit distance 1, or 2 for long words); ties favour the longer word."""
    from rapidfuzz.distance import OSA
    if len(t) < 4 or t in ADDR_CANON or t in STREET_TYPES:
        return t
    best = None
    for w in STREET_TYPES:
        if w[0] != t[0]:
            continue
        d = OSA.distance(t, w)
        if d <= (1 if len(w) <= 5 else 2) and (best is None or d < best[1] or (d == best[1] and len(w) > len(best[0]))):
            best = (w, d)
    return best[0] if best else t


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
    s = fold(_lig(name_main(name)))
    s = _JUNK_PAREN.sub(" ", s)
    s = _LONGNUM.sub(" ", s)
    is_domain = False
    m = _DOMAIN.match(s.strip())
    if m:
        is_domain = True
        s = m.group(1).replace("-", " ")
    s = s.replace("&", " and ").replace("+", " and ").replace("'", "").replace("’", "")
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


def _addr_clean(addr):
    s = fold(_lig(clean_address_raw(addr)))
    s = _NUMERO.sub(" ", s)
    return _NUM_SUFFIX.sub(r"\1 ", s)


def _comp_tokens(comp, tdict):
    """Canonical alpha tokens of one address component (street-type slot typo-repaired)."""
    raw = [t for t in _SPLIT.split(comp.replace("'", " ")) if t]
    for i, t in enumerate(raw):  # the word right after the first house number is the street type in FR
        if t.isdigit():
            if i + 1 < len(raw) and raw[i + 1].isalpha() and is_latin_token(raw[i + 1]):
                raw[i + 1] = fuzzy_street_type(raw[i + 1])
            break
    toks = " ".join(_translit(t, tdict) for t in raw).split()
    out = []
    for t in toks:
        if t.isdigit():
            continue
        # split alnum like 'b3' / 'srno58p' into alpha part only (digits are in nums)
        for u in re.sub(r"\d+", " ", t).split():
            u = ADDR_CANON.get(u, u)
            if u not in ADDR_NOISE:
                out.append(u)
    return out


def addr_tokens(addr, tdict=None):
    """Canonical ASCII token list and normalised number list for an address."""
    s = _addr_clean(addr)
    nums = [n.lstrip("0") or "0" for n in _DIGITS.findall(s)]
    out = []
    for comp in s.split(","):
        out.extend(_comp_tokens(comp, tdict))
    return out, nums


def addr_components(addr, tdict=None):
    """(street tokens, [locality component strings]).

    The street component is the first comma-separated component holding a digit
    (else the first component); every other non-empty component is a locality
    (city, district, state / region ...), kept as a canonical string.
    """
    comps = [c for c in _addr_clean(addr).split(",") if c.strip()]
    if not comps:
        return [], []
    si = next((i for i, c in enumerate(comps) if _DIGITS.search(c)), 0)
    street = _comp_tokens(comps[si], tdict)
    locs = []
    for i, c in enumerate(comps):
        if i != si:
            t = " ".join(_comp_tokens(c, tdict))
            if t and t not in locs:
                locs.append(t)
    return street, locs


def script_of(s):
    """Coarse script id of a string: 0 latin/empty, 1 indic, 2 other."""
    if not s:
        return 0
    if has_indic(s):
        return 1
    if any(ord(c) >= 0x250 and c.isalpha() for c in s):
        return 2
    return 0

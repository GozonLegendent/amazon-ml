#!/usr/bin/env python3
"""Build <team_name>_submission.zip in the layout required by the challenge README.

    <team_name>_submission.zip
    ├── output/
    │   ├── matching_results.tsv
    │   └── candidate_pairs.tsv
    ├── code/
    │   └── business_entity_resolution/   (src/, README.md, requirements.txt, run.sh, ...)
    └── Documentation_template.md

Stdlib only, Python 3.8+. It only reads the inputs; the zip is written to a temporary
file first and renamed into place once it has been re-read and checked.

    python3 build_submission_zip.py --team TEAM \
        --matching   /path/output/matching_results.tsv \
        --candidates /path/output/candidate_pairs.tsv \
        --code-dir   /path/submission/code/business_entity_resolution \
        --doc        /path/submission/Documentation_template.md \
        --out-dir    /path/to/dest            [--force] [--check-subset] [--strict]

--check-subset streams both TSVs and checks every matched id is a candidate of the
same S1 entity. It holds the candidate lists in memory (a few GB for the full test
set), so run it on the GPU box, not on a small laptop.
"""
import argparse
import fnmatch
import os
import re
import sys
import tempfile
import zipfile

CODE_ARC = "code/business_entity_resolution"
REQUIRED_CODE = ["README.md", "requirements.txt", "run.sh", "src/__init__.py"]
HEADERS = {"matching": "source1_entity_id\tmatched_entity_ids",
           "candidates": "source1_entity_id\tcandidate_entity_ids"}
EXCLUDE_DIRS = {"__pycache__", ".git", "snapshots", "logs", "work", "dev_data", "dev_work",
                "dev_output", ".ipynb_checkpoints", ".pytest_cache", ".mypy_cache", "wheels"}
EXCLUDE_FILES = ["*.pyc", "*.pyo", ".DS_Store", "*.log", "*.parquet", "*.npy", "*.zip",
                 "*.pt", "*.bin", "*.safetensors", "*.tsv", "*.swp", "*~"]
MAX_CODE_FILE = 5 * 2**20            # anything bigger under code/ is almost surely data or weights
PLACEHOLDERS = ["[TBD]", "[Your Team Name]", "[List all team members]", "[Date]"]


def die(msg):
    sys.exit(f"ERROR: {msg}")


def check_tsv_header(path, kind):
    if not os.path.isfile(path):
        die(f"{kind} file not found: {path}")
    with open(path, encoding="utf-8", newline="") as f:
        head = f.readline().rstrip("\r\n")
    if head != HEADERS[kind]:
        die(f"{path}: header {head!r} != {HEADERS[kind]!r}")


def check_subset(matching, candidates):
    """Every matched id must also be a candidate of the same S1 entity."""
    cand = {}
    with open(candidates, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.rstrip("\n").partition("\t")
            cand[s1] = frozenset(rest.split(",")) if rest else frozenset()
    bad = n = 0
    with open(matching, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.rstrip("\n").partition("\t")
            if not rest:
                continue
            ids = rest.split(",")
            n += len(ids)
            c = cand.get(s1, frozenset())
            bad += sum(1 for i in ids if i not in c)
    if bad:
        die(f"{bad} of {n} matched ids are not candidates of their S1 entity")
    print(f"  subset check OK: {n} matched ids, all in candidate_pairs.tsv")


def excluded(name, is_dir):
    if is_dir:
        return name in EXCLUDE_DIRS or name.startswith(".")
    return name.startswith(".") or any(fnmatch.fnmatch(name, p) for p in EXCLUDE_FILES)


def code_files(code_dir):
    """(abs_path, arcname) for every file to ship from the code folder, sorted."""
    out = []
    for root, dirs, files in os.walk(code_dir, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not excluded(d, True))
        for fn in sorted(files):
            p = os.path.join(root, fn)
            if excluded(fn, False) or os.path.islink(p):
                continue
            if os.path.getsize(p) > MAX_CODE_FILE:
                die(f"{p} is {os.path.getsize(p) / 2**20:.1f} MB; refusing to ship large files under code/")
            rel = os.path.relpath(p, code_dir).replace(os.sep, "/")
            out.append((p, f"{CODE_ARC}/{rel}"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--team", required=True)
    ap.add_argument("--matching", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--code-dir", required=True)
    ap.add_argument("--doc", required=True)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--force", action="store_true", help="overwrite an existing zip")
    ap.add_argument("--check-subset", action="store_true", help="check matches are a subset of candidates (memory heavy)")
    ap.add_argument("--strict", action="store_true", help="fail (not warn) on doc placeholders / unpinned requirements")
    a = ap.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", a.team):
        die("--team may only contain letters, digits, '_' and '-'")
    code_dir = os.path.realpath(a.code_dir)
    if not os.path.isdir(code_dir):
        die(f"code dir not found: {code_dir}")
    if os.path.basename(code_dir) != "business_entity_resolution":
        print(f"WARNING: code dir is named {os.path.basename(code_dir)!r}; it is stored as {CODE_ARC}/")
    for rel in REQUIRED_CODE:
        if not os.path.isfile(os.path.join(code_dir, rel)):
            die(f"missing required file {os.path.join(code_dir, rel)}")
    if not os.path.isfile(a.doc):
        die(f"documentation not found: {a.doc}")
    check_tsv_header(a.matching, "matching")
    check_tsv_header(a.candidates, "candidates")
    if a.check_subset:
        check_subset(a.matching, a.candidates)

    warnings = []
    with open(a.doc, encoding="utf-8") as f:
        doc = f.read()
    warnings += [f"{a.doc} still contains placeholder {p!r}" for p in PLACEHOLDERS if p in doc]
    with open(os.path.join(code_dir, "requirements.txt"), encoding="utf-8") as f:
        for ln in f:
            ln = ln.split("#", 1)[0].strip()
            if ln and "==" not in ln and not ln.startswith(("-", "--")):
                warnings.append(f"requirements.txt: {ln!r} is not pinned with ==")
    for w in warnings:
        print(f"WARNING: {w}")
    if warnings and a.strict:
        die("--strict: fix the warnings above")

    out_dir = os.path.realpath(a.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, f"{a.team}_submission.zip")
    if os.path.commonpath([dest, code_dir]) == code_dir:
        die("--out-dir must not be inside --code-dir")
    if os.path.exists(dest) and not a.force:
        die(f"{dest} exists; pass --force to overwrite")

    entries = [(os.path.realpath(a.matching), "output/matching_results.tsv"),
               (os.path.realpath(a.candidates), "output/candidate_pairs.tsv")]
    entries += code_files(code_dir)
    entries.append((os.path.realpath(a.doc), "Documentation_template.md"))
    arcs = [arc for _, arc in entries]
    if len(arcs) != len(set(arcs)):
        die("duplicate archive names")

    fd, tmp = tempfile.mkstemp(prefix=".partial_", suffix=".zip", dir=out_dir)
    os.close(fd)
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as zf:
            for src, arc in entries:
                zf.write(src, arc)              # streams from disk; keeps the executable bit of run.sh
        with zipfile.ZipFile(tmp) as zf:        # re-read: CRC of every member, expected names only
            badfile = zf.testzip()
            if badfile:
                die(f"CRC error in {badfile}")
            names = zf.namelist()
            if names != arcs:
                die("archive listing does not match what was written")
            for n in names:
                if n.startswith("/") or ".." in n.split("/") or not (
                        n.startswith(("output/", CODE_ARC + "/")) or n == "Documentation_template.md"):
                    die(f"unexpected archive member {n!r}")
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    total = sum(os.path.getsize(s) for s, _ in entries)
    print(f"wrote {dest} ({os.path.getsize(dest) / 2**20:.1f} MB zipped, {total / 2**20:.1f} MB raw, {len(entries)} files)")
    for _, arc in entries:
        print(f"  {arc}")


if __name__ == "__main__":
    main()

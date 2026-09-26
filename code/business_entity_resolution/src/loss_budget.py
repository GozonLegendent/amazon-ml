"""Diagnostic: split the validation macro F0.5 loss (1 - F) into causes, with oracle what-if gains.

Validation is computed exactly as in decide.tune: fold-0 S1 entities, fold-0 queries, each query
reduced to its argmax S1 (decide.best_per_query), a link kept if p >= t, and macro F0.5 per S1
entity with entities that have no true matches included (decide.f05_macro).

Every true (query, S1) pair of the validation set gets exactly one fate:
  a  not in the blocking candidates (cands/train.parquet)
  b  in the candidates but pruned away (not among the scored pairs)
  c  scored, but the query's argmax is another S1
  d  argmax is the true S1 but p < t
  e  linked correctly
and every wrong predicted link one of:
  f  false link onto an S1 that has true matches
  g  false link onto an S1 with no true matches (a singleton: costs that entity its full point)
gain      = F after repairing only that category (a-d: add the missed true links, f/g: remove the
            false links) minus the current F
gain_last = 1 - F with every other category repaired (what it costs once everything else is right)
c*        = repair c and also drop that query's false link (the argmax had picked a wrong entity)
Gains are not additive (F0.5 is non-linear per entity); repairing a-d, f and g together gives F = 1.
Every gain, also in the slice and band tables, is in overall validation macro F0.5 units, with only
that slice's cases repaired. Slices: S1 country x query address class (addr / noaddr-namesake: no
address and >= 2 training S1 entities in the country share the query's exact ntok, the postfix
namesake rule / noaddr-unique) and query source (src2 / src3). The d, f and g cases are also split
by p band (the argmax probability).
Usage: python -m src.loss_budget --data-dir D --work-dir W [--scored-file F] [--t 0.36,0.75]
"""
import argparse

import polars as pl

from .common import Paths, log, timer
from .decide import best_per_query, f05_macro

MISS = ["a", "b", "c", "d"]
FALSE = ["f", "g"]
FIX = MISS + FALSE
WHAT = {"a": "not in blocking candidates", "b": "in candidates, pruned away", "c": "scored, argmax is another S1",
        "c*": "c + drop that query's false link", "d": "argmax right, p < t", "e": "linked correctly",
        "f": "false link, S1 has true matches", "g": "false link, singleton S1", "all": "repair a-d, f, g"}
PBANDS = [0.0, 0.36, 0.5, 0.75, 0.9, 0.99]
ADDR = ["addr", "noaddr-namesake", "noaddr-unique"]
KEY = ["s1_idx", "q_idx"]
ID = pl.Int64


def pband(col="p"):
    e = pl.lit(f"{PBANDS[-1]:.2f}+")
    for lo, hi in reversed(list(zip(PBANDS[:-1], PBANDS[1:]))):
        e = pl.when(pl.col(col) < hi).then(pl.lit(f"{lo:.2f}-{hi:.2f}")).otherwise(e)
    return e.alias("band")


def _f(tp, npred, ngt):
    """Per-entity F0.5, the same expression as decide.f05_macro."""
    tp, npred, ngt = pl.col(tp), pl.col(npred), pl.col(ngt)
    P, R = tp / npred, tp / ngt
    return (pl.when((ngt == 0) & (npred == 0)).then(1.0).when(tp == 0).then(0.0)
              .otherwise(1.25 * P * R / (0.25 * P + R)))


def gains(u, items, keys, n_ent):
    """Oracle gain per group of `keys`: apply each item's (dtp, dnp) to its entity's counts in u."""
    per = (items.group_by(keys + ["s1_idx"]).agg(pl.len().alias("n"), pl.col("dtp").sum(), pl.col("dnp").sum())
                .join(u, on="s1_idx")
                .with_columns((pl.col("tp") + pl.col("dtp")).alias("tp2"), (pl.col("npred") + pl.col("dnp")).alias("np2")))
    per = per.with_columns((_f("tp2", "np2", "ngt") - pl.col("f")).alias("df"))
    return per.group_by(keys).agg(pl.col("n").sum(), pl.col("s1_idx").n_unique().alias("n_S1"),
                                  (pl.col("df").sum() / n_ent).alias("gain"))


def gain1(u, items, n_ent):
    if items.height == 0:
        return 0.0
    return float(gains(u, items.with_columns(pl.lit(0).alias("_k")), ["_k"], n_ent)["gain"][0])


def f_explicit(pred, gt, universe, items):
    """F0.5 via decide.f05_macro after really editing the link set (the check on the count-based gains)."""
    add = items.filter(pl.col("dtp") == 1).select(KEY)
    drop = items.filter(pl.col("dnp") == -1).select(KEY)
    p2 = pl.concat([pred.select(KEY).join(drop, on=KEY, how="anti"), add])
    return f05_macro(p2, gt, universe)[0]


def load(P, scored_file):
    s1 = pl.read_parquet(P.w("train", "s1.parquet"), columns=["idx", "fold", "country", "ntok"])
    val = s1.filter(pl.col("fold") == 0).select(pl.col("idx").cast(ID).alias("s1_idx"), "country")
    m_ex = s1.group_by("country", "ntok").len("m_ex")   # all training S1 entities: any of them can take the link
    qa = (pl.scan_parquet(P.w("train", "q.parquet")).filter(pl.col("fold") == 0)
            .select("idx", "country", "ntok", "src", "amiss").collect()
            .join(m_ex, on=["country", "ntok"], how="left")
            .select(pl.col("idx").cast(ID).alias("q_idx"), pl.format("src{}", pl.col("src")).alias("src"),
                    pl.when(~pl.col("amiss").fill_null(False)).then(pl.lit("addr"))
                      .when(pl.col("m_ex").fill_null(0) >= 2).then(pl.lit("noaddr-namesake"))
                      .otherwise(pl.lit("noaddr-unique")).alias("addr")))
    del s1, m_ex
    # same gt as decide.tune (true pairs of fold-0 S1 entities)
    gt = (pl.read_parquet(P.w("train", "gt.parquet")).select(pl.col("q_idx").cast(ID), pl.col("s1_idx").cast(ID))
            .join(val.select("s1_idx"), on="s1_idx"))
    if gt.height != gt.unique().height:
        log.warning(f"gt has {gt.height - gt.unique().height} duplicate pairs (decide counts them too)")
    n_offq = gt.join(qa.select("q_idx"), on="q_idx", how="anti").height
    if n_offq:
        log.warning(f"{n_offq} true validation pairs have a query outside fold 0: decide can never link them")

    # argmax per query over the whole scored file with its original dtypes, exactly as decide does
    scored = pl.read_parquet(scored_file, columns=["q_idx", "s1_idx", "p"])
    best = (best_per_query(scored).select(pl.col("q_idx").cast(ID), pl.col("s1_idx").cast(ID), "p")
                                  .join(qa.select("q_idx"), on="q_idx"))
    p_true = (scored.select(pl.col("q_idx").cast(ID), pl.col("s1_idx").cast(ID), "p").join(gt, on=KEY, how="semi")
                    .group_by(KEY).agg(pl.col("p").max().alias("p_true")))
    n_scored = scored.height
    del scored

    def hits(path, name):
        return (pl.scan_parquet(path).select(pl.col("q_idx").cast(ID), pl.col("s1_idx").cast(ID))
                  .join(gt.lazy(), on=KEY, how="semi").unique().collect(engine="streaming")
                  .with_columns(pl.lit(True).alias(name)))

    in_c = hits(P.w("cands", "train.parquet"), "in_cands")
    in_p = hits(P.w("pruned", "train.parquet"), "in_pruned")
    tr = (gt.join(p_true, on=KEY, how="left").join(in_c, on=KEY, how="left").join(in_p, on=KEY, how="left")
            .join(best.select("q_idx", pl.col("s1_idx").alias("best_s1")), on="q_idx", how="left")
            .join(val, on="s1_idx").join(qa, on="q_idx", how="left")
            .with_columns(pl.col("in_cands").fill_null(False), pl.col("in_pruned").fill_null(False)))
    n_mis = tr.filter(pl.col("in_pruned") != pl.col("p_true").is_not_null()).height
    if n_mis:
        log.warning(f"{n_mis} true validation pairs are in pruned/train.parquet xor the scored file: "
                    f"'survived pruning' means 'is in the scored file'")
    log.info(f"scored file {scored_file}: {n_scored} pairs | validation: {val.height} S1 entities, "
             f"{qa.height} fold-0 queries, {gt.height} true pairs (blocking recall {tr['in_cands'].mean():.5f}, "
             f"after pruning {tr['p_true'].is_not_null().mean():.5f})")
    return dict(val=val, qa=qa, gt=gt, best=best, tr=tr)


def report(D, t):
    val, qa, gt, best, tr = D["val"], D["qa"], D["gt"], D["best"], D["tr"]
    universe = val.select("s1_idx")
    pred = best.join(universe, on="s1_idx").filter(pl.col("p") >= t)
    F, u = f05_macro(pred.select(KEY), gt, universe)
    N = u.height

    fate = (pl.when(pl.col("in_pred").fill_null(False)).then(pl.lit("e"))
              .when(pl.col("best_s1") == pl.col("s1_idx")).then(pl.lit("d"))
              .when(pl.col("p_true").is_not_null()).then(pl.lit("c"))
              .when(pl.col("in_cands")).then(pl.lit("b"))
              .otherwise(pl.lit("a")))
    it_true = (tr.join(pred.select(*KEY, pl.lit(True).alias("in_pred")), on=KEY, how="left")
                 .with_columns(fate.alias("cat"))
                 .select(*KEY, "cat", "country", "addr", "src", pl.col("p_true").alias("p"),
                         (pl.col("cat") != "e").cast(ID).alias("dtp"), (pl.col("cat") != "e").cast(ID).alias("dnp")))
    ngt = gt.group_by("s1_idx").len("ngt")
    it_false = (pred.join(gt, on=KEY, how="anti").join(ngt, on="s1_idx", how="left")
                    .join(val, on="s1_idx").join(qa, on="q_idx", how="left")
                    .select(*KEY, pl.when(pl.col("ngt").is_null()).then(pl.lit("g")).otherwise(pl.lit("f")).alias("cat"),
                            "country", "addr", "src", "p", pl.lit(0, ID).alias("dtp"), pl.lit(-1, ID).alias("dnp")))
    items = pl.concat([it_true, it_false])
    fix = items.filter(pl.col("cat").is_in(FIX))

    # ---- checks
    cnt = dict(items.group_by("cat").len().iter_rows())
    part_ok = (sum(cnt.get(k, 0) for k in "abcde") == gt.height and it_true.height == gt.height
               and it_true.select(KEY).unique().height == gt.unique().height)
    link_ok = cnt.get("e", 0) + cnt.get("f", 0) + cnt.get("g", 0) == pred.height
    F_all_x = f_explicit(pred, gt, universe, fix)
    F_all = F + gain1(u, fix, N)

    # ---- overall causes
    rows, max_diff = [], abs(F_all - F_all_x)
    c_q = fix.filter(pl.col("cat") == "c").select("q_idx")
    for k in ["a", "b", "c", "c*", "d", "e", "f", "g", "all"]:
        if k == "e":
            sub = items.filter(pl.col("cat") == "e")
            rows.append((k, WHAT[k], sub.height, sub["s1_idx"].n_unique(), None, None))
            continue
        if k == "all":
            sub = fix
        elif k == "c*":
            sub = pl.concat([fix.filter(pl.col("cat") == "c"),
                             fix.filter(pl.col("cat").is_in(FALSE)).join(c_q, on="q_idx", how="semi")])
        else:
            sub = fix.filter(pl.col("cat") == k)
        g = gain1(u, sub, N)
        max_diff = max(max_diff, abs(F + g - f_explicit(pred, gt, universe, sub)))
        if k in FIX:
            g_last = F_all - (F + gain1(u, fix.filter(pl.col("cat") != k), N))
        else:
            g_last = F_all - F if k == "all" else None
        rows.append((k, WHAT[k], sub.height, sub["s1_idx"].n_unique(), g, g_last))
    causes = pl.DataFrame(rows, schema=["cause", "what", "n", "n_S1", "gain", "gain_last"], orient="row")
    causes = causes.with_columns(pl.col("gain").round(5), pl.col("gain_last").round(5))

    # ---- header
    uc = u.join(val, on="s1_idx")
    sing = u.filter(pl.col("ngt") == 0)
    print(f"\n===== t = {t:g} | VALIDATION macro F0.5 = {F:.5f} (decide.f05_macro) | loss 1-F = {1 - F:.5f} =====")
    print(f"{N} S1 entities ({sing.height} singletons, F {sing['f'].mean() if sing.height else float('nan'):.4f}) | "
          f"{gt.height} true pairs | {pred.height} links ({cnt.get('f', 0) + cnt.get('g', 0)} false)")
    for c, g in sorted(uc.group_by("country"), key=lambda x: x[0][0]):
        print(f"  country {c[0]}: F0.5 {g['f'].mean():.4f} over {g.height} | its loss in overall F units "
              f"{(1 - g['f']).sum() / N:.5f}")
    print(f"checks: a-e partition the {gt.height} true pairs: {'OK' if part_ok else 'FAIL'} | e+f+g = {pred.height} "
          f"links: {'OK' if link_ok else 'FAIL'} | all repairs -> F = {F_all_x:.6f} (f05_macro), {F_all:.6f} "
          f"(counts) | count gains vs f05_macro max diff {max_diff:.1e}")
    if not (part_ok and link_ok and abs(F_all_x - 1) < 1e-9 and max_diff < 1e-9):
        log.warning("a consistency check FAILED: read the tables with care")
    print(causes)

    # ---- slices: S1 country x (query address class | query source)
    ctry = ["all"] + sorted(val["country"].unique().to_list())
    srcs = sorted(qa["src"].unique().to_list())
    sl = ["all"] + ADDR + srcs
    ex = (items.with_columns(pl.concat_list(pl.lit("all"), pl.col("country")).alias("ctry"),
                             pl.concat_list(pl.lit("all"), pl.col("addr"), pl.col("src")).alias("slice"))
               .explode("ctry", empty_as_null=True).explode("slice", empty_as_null=True))
    skel = pl.DataFrame({"ctry": ctry}).with_row_index("_o1").join(
        pl.DataFrame({"slice": sl}).with_row_index("_o2"), how="cross")
    tot = ex.group_by("ctry", "slice").agg(pl.col("cat").is_in(list("abcde")).sum().alias("n_true"),
                                           pl.col("cat").is_in(["e"] + FALSE).sum().alias("n_link"))
    out = skel.join(tot, on=["ctry", "slice"], how="left")
    g = gains(u, ex.filter(pl.col("cat").is_in(FIX)), ["ctry", "slice", "cat"], N)
    for k in FIX:
        part = g.filter(pl.col("cat") == k).select("ctry", "slice", pl.col("n").alias(f"n_{k}"),
                                                   pl.col("gain").round(5).alias(f"g_{k}"))
        out = out.join(part, on=["ctry", "slice"], how="left")
    slices = (out.fill_null(0).filter((pl.col("n_true") + pl.col("n_link")) > 0)
                 .sort("_o1", "_o2").drop("_o1", "_o2").rename({"ctry": "country"}))
    print(f"--- by slice (t = {t:g}): n_k = cases, g_k = overall-F gain from repairing only that slice's k ---")
    print(slices)

    # ---- p bands of the d / f / g cases
    bx = (items.filter(pl.col("cat").is_in(["d"] + FALSE)).with_columns(pband("p"))
               .with_columns(pl.concat_list(pl.lit("all"), pl.col("country")).alias("ctry")).explode("ctry", empty_as_null=True))
    bands = [f"{lo:.2f}-{hi:.2f}" for lo, hi in zip(PBANDS[:-1], PBANDS[1:])] + [f"{PBANDS[-1]:.2f}+"]
    skel = pl.DataFrame({"ctry": ctry}).with_row_index("_o1").join(
        pl.DataFrame({"band": bands}).with_row_index("_o2"), how="cross")
    out = skel
    if bx.height:
        g = gains(u, bx, ["ctry", "band", "cat"], N)
    else:
        g = pl.DataFrame(schema={"ctry": pl.Utf8, "band": pl.Utf8, "cat": pl.Utf8, "n": pl.UInt32, "gain": pl.Float64})
    for k in ["d"] + FALSE:
        part = g.filter(pl.col("cat") == k).select("ctry", "band", pl.col("n").alias(f"n_{k}"),
                                                   pl.col("gain").round(5).alias(f"g_{k}"))
        out = out.join(part, on=["ctry", "band"], how="left")
    by_band = (out.fill_null(0).filter((pl.col("n_d") + pl.col("n_f") + pl.col("n_g")) > 0)
                  .sort("_o1", "_o2").drop("_o1", "_o2").rename({"ctry": "country"}))
    print(f"--- d / f / g by p band of the argmax (t = {t:g}) ---")
    print(by_band)
    return {"F": F, "gain": {r[0]: r[4] for r in rows}, "causes": causes, "slices": slices, "bands": by_band,
            "ok": part_ok and link_ok and abs(F_all_x - 1) < 1e-9 and max_diff < 1e-9}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--scored-file", default=None, help="train scores parquet to evaluate (default: work/scored/train.parquet)")
    ap.add_argument("--t", default="0.75", help="decision threshold(s), comma separated, e.g. 0.36,0.75")
    args = ap.parse_args()
    P = Paths(args.data_dir, args.work_dir)
    pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(250)
    pl.Config.set_tbl_hide_column_data_types(True); pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_fmt_str_lengths(40)
    ts = [float(x) for x in args.t.split(",") if x.strip()]
    with timer("load validation fold"):
        D = load(P, args.scored_file or P.w("scored", "train.parquet"))
    with timer(f"loss budget at t = {ts}"):
        for t in ts:
            report(D, t)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Recompute the manuscript's fNIRS statistics from the quantified CSVs (data/) and compare with the reported values."""
from __future__ import annotations
import os, sys, io
import numpy as np, pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
L = lambda study, *p: pd.read_csv(os.path.join(DATA, study, *p), encoding="utf-8-sig")

_buf = io.StringIO()
def say(s=""):
    print(s); _buf.write(s + "\n")

RESULTS = []
def check(label, got, want, tol, unit=""):
    """Compare got with want within tol; if want is None, only record the value."""
    if want is None:
        RESULTS.append((label, got, None, None)); say(f"    {label:<46} {got}"); return
    ok = abs(float(got) - float(want)) <= tol
    RESULTS.append((label, got, want, ok))
    mark = "OK " if ok else "!! "
    say(f"  {mark}{label:<46} 재현 {got:>10.4g}{unit}   원고 {want:>10.4g}{unit}")
    return ok

def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n); prev = 1.0
    for i in range(n - 1, -1, -1):
        prev = min(prev, p[o[i]] * n / (i + 1)); q[o[i]] = prev
    return q

def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b)); a, b = a[m], b[m]; d = b - a
    t, p = stats.ttest_rel(b, a)
    return dict(n=len(d), t=t, p=p, delta=d.mean(), dz=d.mean() / d.std(ddof=1),
                same=int((np.sign(d) == np.sign(d.mean())).sum()), a=a.mean(), b=b.mean())

def onesample(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    t, p = stats.ttest_1samp(x, 0); se = x.std(ddof=1) / np.sqrt(len(x))
    lo, hi = stats.t.interval(.95, len(x) - 1, x.mean(), se)
    return dict(n=len(x), mean=x.mean(), lo=lo, hi=hi, t=t, p=p,
                dz=x.mean() / x.std(ddof=1), same=int((np.sign(x) == np.sign(x.mean())).sum()))


# =====================================================================
say("=" * 78); say("STUDY 1"); say("=" * 78)

say("\n[1-2] fNIRS ch01 HbO (visit 2) — 조건별 beta 및 대비")
f = L("study1", "quantified", "study1_fnirs_glm_long.csv")
p2 = f[(f.visit == "V2") & (f.chromophore == "HbO") & (f.channel == "ch01")] \
        .pivot_table(index="id_num", columns="condition", values="beta")
for c, want in [("shortform1", -0.093), ("brainpt", -0.023), ("shortform2", 0.110), ("task", 0.180)]:
    check(f"beta {c}", p2[c].mean(), want, .001)
con = {}
for a, b2, lab, W in [("shortform1", "task", "SF1 vs Task", (0.273, 4.42, .002, 1.47, 8)),
                      ("shortform1", "shortform2", "SF1 vs SF2", (0.203, 3.06, .02, 1.02, 8)),
                      ("brainpt", "task", "BrainPT vs Task", (0.202, 2.52, .04, 0.84, None)),
                      ("shortform1", "brainpt", "SF1 vs BrainPT", (0.071, 1.05, .32, 0.35, None))]:
    r = paired(p2[a], p2[b2]); con[lab] = r
    say(f"    · {lab}")
    check("  Δβ", r["delta"], W[0], .002); check("  t", r["t"], W[1], .02)
    check("  P", r["p"], W[2], .006); check("  dz", r["dz"], W[3], .02)
    if W[4]: check("  동일방향 인원", r["same"], W[4], 0)
q = bh([con[k]["p"] for k in con])
for k, v, want in zip(con, q, [.009, .03, .048, .32]):
    check(f"FDR q — {k}", v, want, .006)

# =====================================================================
say(""); say("=" * 78); say("STUDY 2"); say("=" * 78)

say("\n[2-6] fNIRS ANCOVA family — 보정 후 생존 없음 (504개 ANCOVA family 중 이 부분)")
fam = {'study2_fnirs_full_channel_ancova.csv': 60, 'study2_hrf_ancova.csv': 72}
tot = 0; mins = []
for fn, n in fam.items():
    d2 = L("study2", "neural", fn); tot += len(d2)
    check(f"{fn[:38]:<38} 행 수", len(d2), n, 0)
    qc = [c for c in d2.columns if "fdr" in c.lower()]
    if qc: mins.append(d2[qc].min().min())
check("family 합계", tot, 132, 0)
check("최소 q ≥ .11 (504개 전체 최소 q = .11)", int(min(mins) >= .105), 1, 0)

# =====================================================================
say(""); say("=" * 78)
n_ok = sum(1 for _, _, w, ok in RESULTS if w is not None and ok)
n_bad = sum(1 for _, _, w, ok in RESULTS if w is not None and not ok)
say(f"판정: 대조 {n_ok + n_bad}건 중 일치 {n_ok}, 불일치 {n_bad}")
if n_bad:
    say("\n불일치 항목:")
    for lab, got, want, ok in RESULTS:
        if want is not None and not ok:
            say(f"  - {lab}: 재현 {got:.4g} vs 원고 {want:.4g}")
say("=" * 78)

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "REPRODUCTION_OUTPUT_fnirs.txt"),
          "w", encoding="utf-8") as fh:
    fh.write(_buf.getvalue())
sys.exit(1 if n_bad else 0)

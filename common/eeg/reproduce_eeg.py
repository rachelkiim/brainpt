# -*- coding: utf-8 -*-
"""Recompute the manuscript's EEG statistics from the quantified CSVs (data/) and compare with the reported values."""
from __future__ import annotations
import os, sys, io, itertools
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

BANDS = ["Delta", "Theta", "Alpha", "Beta"]

# =====================================================================
say("=" * 78); say("STUDY 1"); say("=" * 78)

say("\n[1-1] 시청 중 대역 파워 (visit 2, shortform 1-3 평균, baseline 대비)")
say("      원고: alpha −1.72 dB, 95% CI −2.97~−0.46, t8=−3.15, P=.01, dz=−1.05, 8/9, q=.054")
w = L("study1", "quantified", "study1_v2_individual_long.csv") \
    .groupby(["participant", "condition", "band"]).dB.mean().unstack("condition")
sf = w[["shortform1", "shortform2", "shortform3"]].mean(axis=1)
ps = []
for b in BANDS:
    r = onesample(sf[sf.index.get_level_values("band") == b].values); ps.append(r["p"])
    if b == "Alpha":
        check("alpha 평균 (dB)", r["mean"], -1.72, .01)
        check("alpha CI 하한", r["lo"], -2.97, .01); check("alpha CI 상한", r["hi"], -0.46, .01)
        check("alpha t", r["t"], -3.15, .01); check("alpha P", r["p"], .0135, .002)
        check("alpha dz", r["dz"], -1.05, .01); check("alpha 동일방향 인원", r["same"], 8, 0)
    else:
        say(f"      {b:<6} {r['mean']:+.3f} dB  CI[{r['lo']:+.2f}, {r['hi']:+.2f}]  P={r['p']:.3f}")
check("alpha BH q (4대역)", bh(ps)[BANDS.index("Alpha")], .054, .002)

say("\n[1-3] 대안설명 대조 분석 (visit 1, n=8)")
ap = L("study1", "neural", "aperiodic_pilot_results.csv")
pv = ap.pivot_table(index="participant", columns="condition", values=["offset", "exponent", "artifact_var"])
for k, wd, wp in [("offset", -0.072, .62), ("exponent", 0.039, .73), ("artifact_var", -0.044, .31)]:
    a = pv[k]; c0, c1 = list(a.columns)
    t, p = stats.ttest_rel(a[c1], a[c0])
    check(f"{k} Δ", (a[c1] - a[c0]).mean(), wd, .001); check(f"{k} P", p, wp, .01)
av, off, exp = pv["artifact_var"], pv["offset"], pv["exponent"]
c0, c1 = list(off.columns)
check("r(artifact, offset)", np.corrcoef(av[c1] - av[c0], off[c1] - off[c0])[0, 1], .10, .01)
check("r(artifact, exponent)", np.corrcoef(av[c1] - av[c0], exp[c1] - exp[c0])[0, 1], .11, .01)

say("\n[1-4] 채널 수준 (visit 1, 미보정)")
v = L("study1", "neural", "v1_channel_significance.csv")
for ch, bd, want in [("F4", "Alpha", .004), ("AF4", "Beta", .005), ("FC6", "Beta", .008)]:
    r = v[(v.condition == "shortform") & (v.band == bd) & (v.channel == ch)]
    check(f"{ch} {bd} P", float(r.p.iloc[0]), want, .0006)
check("FDR 통과 채널 수", int(v.sig_fdr.sum()), 0, 0)

say("\n[1-5] 시청 중 시간 경과 — 구간 평균과 정확 순열검정")
c = L("study1", "quantified", "study1_v1v2_combined.csv")
segmap = {"shortform_p1": 1, "shortform_p2": 2, "shortform_p3": 3,
          "shortform1": 1, "shortform2": 2, "shortform3": 3}
d = c[c.condition.isin(segmap)].copy(); d["seg"] = d.condition.map(segmap)
M = d.groupby(["band", "participant", "visit", "seg"]).dB.mean().unstack("seg")
PIDS = sorted(d.participant.unique()); N = len(PIDS)
sc = np.array([1., 2., 3.]) - 2.; DEN = (sc ** 2).sum()
SIGNS = np.array(list(itertools.product([1, -1], repeat=N)), float)
say(f"      순열 공간: 2^{N} = {len(SIGNS)} (전수 열거, 몬테카를로 오차 0)")
WANT_SEG = {"Delta": (1.65, -1.82, -0.75, -1.03), "Theta": (0.97, -1.59, -1.07, -0.79),
            "Alpha": (-0.90, -2.37, -1.72, -1.53), "Beta": (-0.33, -1.26, -0.86, -0.68)}
WANT_P = {"Delta": .02, "Theta": .04, "Alpha": .13, "Beta": .11}
exact = {}
for b in BANDS:
    sub = M.loc[b]
    v1 = np.array([sub.loc[(p, "V1")].values for p in PIDS], float)
    v2 = np.array([sub.loc[(p, "V2")].values for p in PIDS], float)
    say(f"    · {b}")
    for lab, arr, idx, want in [("V1 1구간", v1, 0, WANT_SEG[b][0]), ("V1 3구간", v1, 2, WANT_SEG[b][1]),
                                ("V2 1구간", v2, 0, WANT_SEG[b][2]), ("V2 3구간", v2, 2, WANT_SEG[b][3])]:
        check(f"  {lab}", np.nanmean(arr[:, idx]), want, .01)
    dif = (v2 @ sc / DEN) - (v1 @ sc / DEN)
    obs = dif.mean(); null = SIGNS @ dif / N
    pex = np.sum(np.abs(null) >= abs(obs)) / len(SIGNS)
    exact[b] = pex
    check("  정확 순열 P", pex, WANT_P[b], .005)
check("순열 P 의 BH 최소 q", bh([exact[b] for b in BANDS]).min(), .07, .005)

# =====================================================================
say(""); say("=" * 78); say("STUDY 2"); say("=" * 78)

e = L("study2", "quantified", "study2_eeg_combined.csv")
v1 = e[e.visit == "V1"]
wv = v1.groupby(["participant", "band", "condition"]).dB.mean().unstack("condition")

say("\n[2-1] 개입 전(visit 1) 조건별 파워 및 조건 대비")
WANT_M = {"Delta": (-1.83, -1.35, 0.60, .018), "Theta": (-0.54, 0.49, 1.86, .002),
          "Alpha": (-1.05, -0.39, 0.41, .035), "Beta": (-0.86, 0.08, 0.47, .064)}
for b in BANDS:
    s = wv[wv.index.get_level_values("band") == b]
    say(f"    · {b}")
    for c2, want in [("shortform", WANT_M[b][0]), ("brainpt", WANT_M[b][1]), ("task", WANT_M[b][2])]:
        check(f"  {c2} 평균", s[c2].mean(), want, .01)
    r = paired(s["shortform"], s["task"])
    check("  sf vs task P", r["p"], WANT_M[b][3], .001)
    check("  sf vs task n", r["n"], 17, 0)
check("shortform 단독 n", int(wv[wv.index.get_level_values('band') == 'Delta']['shortform'].notna().sum()), 18, 0)

say("\n[2-2] baseline 대비 개별 구별 가능한 3개 셀 (95% CI)")
for cond, b, lo_w, hi_w in [("shortform", "Delta", -3.15, -0.52), ("shortform", "Alpha", -1.96, -0.14),
                            ("task", "Theta", 0.40, 3.33)]:
    r = onesample(wv[wv.index.get_level_values("band") == b][cond].dropna().values)
    check(f"{cond} {b} CI 하한", r["lo"], lo_w, .01); check(f"{cond} {b} CI 상한", r["hi"], hi_w, .01)

say("\n[2-3] 사전지정 4쌍 연결성 (visit 1, BrainPT − shortform)")
cn = L("study2", "quantified", "study2_conn_allpairs.csv")
KEEP = set(e.participant.unique())
spec = ["F3-P7", "AF3-P7", "P8-F4", "P8-AF4"]
sel = [p for p in cn.pair.unique() if p in spec or "-".join(p.split("-")[::-1]) in spec]
fp = cn[(cn.visit == "V1") & cn.pair.isin(sel) & cn.initials.isin(KEEP)]
for b, wdz in [("Beta", 2.31), ("Theta", 1.09), ("Alpha", 1.08)]:
    s = fp[fp.band == b].groupby(["initials", "condition"]).wpli.mean().unstack("condition")
    r = onesample((s["brainpt"] - s["shortform"]).dropna().values)
    check(f"{b} dz", r["dz"], wdz, .01)
    if b == "Beta": check("Beta 동일방향 인원", r["same"], 18, 0)

say("\n[2-4] 91쌍 중 baseline 보다 높은 쌍 수 (visit 1)")
WANT_C = {"Beta": dict(brainpt=91, shortform=0, task=9), "Theta": dict(brainpt=86, shortform=4, task=33),
          "Alpha": dict(brainpt=14, shortform=0, task=1)}
c18 = cn[cn.initials.isin(KEEP)]
for b in ["Beta", "Theta", "Alpha"]:
    piv = c18[(c18.visit == "V1") & (c18.band == b)].groupby(["pair", "condition"]).wpli.mean().unstack("condition")
    for c2, want in WANT_C[b].items():
        check(f"{b} {c2}", int((piv[c2] > piv["baseline"]).sum()), want, 0)

say("\n[2-5] 개입 후 연결성 (BrainPT 조건, Beta)")
grp = e[["participant", "group"]].drop_duplicates().set_index("participant").group.to_dict()
cn2 = c18.copy(); cn2["group"] = cn2.initials.map(grp)
def carry(pairs):
    b = cn2[(cn2.band == "Beta") & cn2.pair.isin(pairs)]
    gl = b.groupby(["initials", "group", "visit", "condition"]).wpli.mean().reset_index()
    s = gl[gl.condition == "brainpt"].pivot_table(index=["initials", "group"], columns="visit",
                                                  values="wpli").reset_index().dropna(subset=["V1", "V2"])
    out = {}
    for g in ["control", "experimental"]:
        sub = s[s.group == g]; dd = (sub.V2 - sub.V1).values
        out[g] = (dd.mean(), int((dd > 0).sum()), len(dd))
    X = np.column_stack([np.ones(len(s)), (s.group == "experimental").astype(float), s.V1.values])
    bb, *_ = np.linalg.lstsq(X, s.V2.values, rcond=None)
    r = s.V2.values - X @ bb; dof = len(s) - 3
    cov = (r @ r / dof) * np.linalg.pinv(X.T @ X)
    t = bb[1] / np.sqrt(cov[1, 1])
    return out, 2 * stats.t.sf(abs(t), dof)
for lab, pairs, W in [("사전지정 4쌍", sel, (0.034, 0.010, .09)),
                      ("전역 91쌍", list(cn.pair.unique()), (0.026, 0.008, .07))]:
    out, pa = carry(pairs); say(f"    · {lab}")
    check("  실험군 Δ", out["experimental"][0], W[0], .001)
    check("  대조군 Δ", out["control"][0], W[1], .001)
    check("  실험군 동일방향", out["experimental"][1], 9, 0)
    check("  ANCOVA P", pa, W[2], .006)

say("\n[2-6] EEG ANCOVA family — 보정 후 생존 없음 (504개 ANCOVA family 중 이 부분)")
fam = {'study2_eeg_exhaustive.csv': 168, 'study2_conn_region_ancova.csv': 180, 'study2_entropy_ancova.csv': 24}
tot = 0; mins = []
for fn, n in fam.items():
    d2 = L("study2", "neural", fn); tot += len(d2)
    check(f"{fn[:38]:<38} 행 수", len(d2), n, 0)
    qc = [c for c in d2.columns if "fdr" in c.lower()]
    if qc: mins.append(d2[qc].min().min())
check("family 합계", tot, 372, 0)
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

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "REPRODUCTION_OUTPUT_eeg.txt"),
          "w", encoding="utf-8") as fh:
    fh.write(_buf.getvalue())
sys.exit(1 if n_bad else 0)

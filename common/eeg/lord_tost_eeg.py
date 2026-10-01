# -*- coding: utf-8 -*-
"""Study 2 EEG (wPLI, band power): group difference in V1->V2 change, change-score t-test vs ANCOVA (Lord's paradox) and TOST equivalence."""
import os, io
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
L = lambda study, *p: pd.read_csv(os.path.join(DATA, study, *p), encoding="utf-8-sig")

_buf = io.StringIO()
def say(s=""):
    print(s); _buf.write(s + "\n")

# Group assignment (from the EEG sample)
eeg = L("study2", "quantified", "study2_eeg_combined.csv")
GRP = eeg[["participant", "group"]].drop_duplicates().set_index("participant").group.to_dict()
KEEP = set(GRP)

def ancova(df):
    """df: initials, group, V1, V2 → (coef, p, n)"""
    d = df.dropna(subset=["V1", "V2"])
    X = np.column_stack([np.ones(len(d)), (d.group == "experimental").astype(float), d.V1.values])
    b, *_ = np.linalg.lstsq(X, d.V2.values, rcond=None)
    r = d.V2.values - X @ b; dof = len(d) - 3
    cov = (r @ r / dof) * np.linalg.pinv(X.T @ X)
    t = b[1] / np.sqrt(cov[1, 1])
    return b[1], 2 * stats.t.sf(abs(t), dof), len(d)

def change_test(df):
    """Welch t-test on the change (V2 - V1) between groups."""
    d = df.dropna(subset=["V1", "V2"]).copy()
    d["chg"] = d.V2 - d.V1
    ex = d[d.group == "experimental"].chg.values
    co = d[d.group == "control"].chg.values
    t, p = stats.ttest_ind(ex, co, equal_var=False)
    sp = np.sqrt(((len(ex) - 1) * ex.var(ddof=1) + (len(co) - 1) * co.var(ddof=1)) / (len(ex) + len(co) - 2))
    return ex.mean() - co.mean(), t, p, (ex.mean() - co.mean()) / sp, len(ex), len(co), ex, co

def tost(ex, co, bound_d, label):
    """Two-sample TOST; bound = bound_d x pooled SD."""
    n1, n2 = len(ex), len(co)
    s1, s2 = ex.var(ddof=1), co.var(ddof=1)
    diff = ex.mean() - co.mean()
    se = np.sqrt(s1 / n1 + s2 / n2)
    dof = (s1 / n1 + s2 / n2) ** 2 / ((s1 / n1) ** 2 / (n1 - 1) + (s2 / n2) ** 2 / (n2 - 1))
    sp = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    delta = bound_d * sp
    p1 = stats.t.sf((diff + delta) / se, dof)      # H01: diff <= -delta
    p2 = stats.t.cdf((diff - delta) / se, dof)     # H02: diff >= +delta
    return dict(index=label, bound_d=bound_d, bound_raw=delta, diff=diff, se=se, df=dof,
                p_lower=p1, p_upper=p2, p_tost=max(p1, p2),
                equivalence="established" if max(p1, p2) < .05 else "NOT established")
# Build indices
say("=" * 82); say("지표 구성"); say("=" * 82)
IDX = {}

cn = L("study2", "quantified", "study2_conn_allpairs.csv")
cn = cn[cn.initials.isin(KEEP)].copy()
cn["group"] = cn.initials.map(GRP)
spec = ["F3-P7", "AF3-P7", "P8-F4", "P8-AF4"]
SEL4 = [p for p in cn.pair.unique() if p in spec or "-".join(p.split("-")[::-1]) in spec]
ALLP = list(cn.pair.unique())
for scope, pairs in [("wPLI_4pair", SEL4), ("wPLI_global", ALLP)]:
    sub = cn[cn.pair.isin(pairs)]
    for band in sorted(sub.band.unique()):
        for cond in ["brainpt", "shortform", "task"]:
            s = sub[(sub.band == band) & (sub.condition == cond)]
            w = s.groupby(["initials", "group", "visit"]).wpli.mean().unstack("visit").reset_index()
            if {"V1", "V2"} <= set(w.columns):
                IDX[f"{scope}_{band}_{cond}"] = w

pw = eeg.groupby(["participant", "group", "visit", "condition", "band"]).dB.mean().reset_index()
for band in ["Delta", "Theta", "Alpha", "Beta"]:
    for cond in ["brainpt", "shortform", "task"]:
        s = pw[(pw.band == band) & (pw.condition == cond)]
        w = s.pivot_table(index=["participant", "group"], columns="visit", values="dB").reset_index()
        w = w.rename(columns={"participant": "initials"})
        if {"V1", "V2"} <= set(w.columns):
            IDX[f"power_{band}_{cond}"] = w

say(f"  구성된 지표 {len(IDX)}개")

# Change-score t-test vs ANCOVA
say(""); say("=" * 82); say("3) Lord 규칙 — 변화량 t검정 대 ANCOVA"); say("=" * 82)
rows = []
for name, w in sorted(IDX.items()):
    diff, t, p_chg, dd, ne, nc, ex, co = change_test(w)
    coef, p_anc, n = ancova(w)
    sig_c, sig_a = p_chg < .05, p_anc < .05
    same_sign = bool(np.sign(diff) == np.sign(coef))
    if sig_c != sig_a:
        verdict = "DISAGREE_significance"     # not adopted
    elif sig_c and sig_a and not same_sign:
        verdict = "DISAGREE_sign"             # both significant, opposite sign
    elif not sig_c and not sig_a and not same_sign:
        verdict = "agree_null_signflip"       # both n.s., sign differs
    else:
        verdict = "agree"
    rows.append(dict(index=name, n=n, n_exp=ne, n_con=nc,
                     change_diff=diff, change_t=t, change_p=p_chg, change_d=dd,
                     ancova_coef=coef, ancova_p=p_anc,
                     sig_change=sig_c, sig_ancova=sig_a, same_sign=same_sign, verdict=verdict))
lord = pd.DataFrame(rows)
lord.to_csv(os.path.join(HERE, "lord_change_vs_ancova.csv"), index=False, encoding="utf-8-sig")
dis = lord[lord.verdict.str.startswith("DISAGREE")]
flip = lord[lord.verdict == "agree_null_signflip"]
say(f"  검정 지표 {len(lord)}개")
say(f"  변화량 검정 유의 {int(lord.sig_change.sum())}개 · ANCOVA 유의 {int(lord.sig_ancova.sum())}개")
say(f"  실질 불일치(사전 규정 발동): {len(dis)}개 | 둘 다 비유의·부호만 상이: {len(flip)}개")
if len(dis):
    say("\n  !! 사전 규정상 미채택 대상")
    for _, r in dis.iterrows():
        say(f"     {r['index']:<34} 변화량 p={r.change_p:.4f} (Δ={r.change_diff:+.4f}) | "
            f"ANCOVA p={r.ancova_p:.4f} (coef={r.ancova_coef:+.4f})  [{r.verdict}]")
if len(flip):
    say("\n  (참고) 둘 다 비유의이며 부호만 다른 항목 — 결론은 동일")
    for _, r in flip.iterrows():
        say(f"     {r['index']:<34} 변화량 p={r.change_p:.3f} | ANCOVA p={r.ancova_p:.3f}")
say("\n  주요 지표 발췌")
for k in ["wPLI_4pair_Beta_brainpt", "wPLI_global_Beta_brainpt"]:
    if k in set(lord["index"]):
        r = lord[lord["index"] == k].iloc[0]
        say(f"    {k:<28} 변화량 Δ={r.change_diff:+.4f} p={r.change_p:.3f} | ANCOVA p={r.ancova_p:.3f}")

# ================= 4) TOST =================
say(""); say("=" * 82); say("4) TOST — 집단 간 변화량 차이의 등가성"); say("=" * 82)
say("  등가 한계 두 가지: d=1.42 (원고에 기재된 최소검출효과) · d=0.50 (관례값)")
trows = []
TARGET = [k for k in IDX if k.startswith("wPLI_4pair")]
for name in sorted(TARGET):
    _, _, _, _, _, _, ex, co = change_test(IDX[name])
    for bd in (1.42, 0.50):
        trows.append(tost(ex, co, bd, name))
tostdf = pd.DataFrame(trows)
tostdf.to_csv(os.path.join(HERE, "tost_results.csv"), index=False, encoding="utf-8-sig")
say(f"\n  {'지표':<30}{'한계 d':>7}{'차이':>10}{'p_TOST':>10}  판정")
for _, r in tostdf.iterrows():
    say(f"  {r['index']:<30}{r.bound_d:>7.2f}{r['diff']:>10.4f}{r.p_tost:>10.3f}  {r.equivalence}")
est = tostdf[tostdf.equivalence == "established"]
say(f"\n  등가성 확립 {len(est)}건 / 전체 {len(tostdf)}건")
say("  ※ 큰 p_TOST 는 '차이 없음'이 아니라 '등가성 미확립'을 뜻한다.")

say("")
say("=" * 82)
say("산출: lord_change_vs_ancova.csv · tost_results.csv")
say("=" * 82)
open(os.path.join(HERE, "lord_tost_eeg_output.txt"), "w", encoding="utf-8").write(_buf.getvalue())

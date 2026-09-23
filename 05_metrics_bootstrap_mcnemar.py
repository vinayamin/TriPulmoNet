"""Metrics, stratified bootstrap and McNemar, as used for the locked numbers.
Bootstrap: B=2000, rng=np.random.default_rng(42); each replicate resamples WITH replacement within each true
class (order TB, Pneumonia, Normal) keeping class sizes fixed; CI = 2.5th/97.5th np.percentile (linear).
Paired comparisons use the SAME replicate indices for both models (final C-vs-D analysis: fresh rng per comparison).
McNemar: per-image paired correctness; exact two-sided binomial if discordant n < 25, else chi-square with
continuity correction (|b-c|-1)^2/(b+c), 1 df.
"""
import numpy as np
from scipy.stats import binomtest, chi2
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
def metrics(y,P):
    p=P.argmax(1)
    return dict(acc=accuracy_score(y,p),bal=balanced_accuracy_score(y,p),mf1=f1_score(y,p,average="macro"),
                auc=roc_auc_score(y,P,multi_class="ovr",average="macro"))
def strat_indices(y,B=2000,seed=42):
    rng=np.random.default_rng(seed); cls=[np.where(y==c)[0] for c in range(3)]
    for _ in range(B): yield np.concatenate([rng.choice(ic,size=len(ic),replace=True) for ic in cls])
def bootstrap_ci(y,P):
    d={k:[] for k in ["acc","bal","mf1","auc"]}
    for idx in strat_indices(y):
        m=metrics(y[idx],P[idx]); [d[k].append(m[k]) for k in d]
    return {k:(float(np.percentile(v,2.5)),float(np.percentile(v,97.5))) for k,v in d.items()}
def paired_diff_ci(y,P1,P2):
    d={k:[] for k in ["acc","bal","mf1","auc"]}
    for idx in strat_indices(y):
        a,b=metrics(y[idx],P1[idx]),metrics(y[idx],P2[idx]); [d[k].append(a[k]-b[k]) for k in d]
    return {k:(float(np.percentile(v,2.5)),float(np.percentile(v,97.5))) for k,v in d.items()}
def mcnemar(ok1,ok2):
    b=int((ok1&~ok2).sum()); c=int((~ok1&ok2).sum()); n=b+c
    if n==0: return dict(b=b,c=c,p=1.0,test="none")
    if n<25: return dict(b=b,c=c,p=binomtest(min(b,c),n,0.5).pvalue,test="exact binomial")
    return dict(b=b,c=c,p=float(chi2.sf((abs(b-c)-1)**2/n,1)),test="chi2 continuity-corrected")
def screening_offsets(P,offsets=(0.75,0.5,0.0),eps=1e-12):
    """Secondary operating point: argmax(log(p+eps)+offset); offsets tuned on TTA-matched OOF predictions of
    Variant D by grid search maximizing OOF balanced accuracy (coarse grid -2..2 step 0.25, fine grid step 0.025)."""
    return (np.log(P+eps)+np.asarray(offsets)).argmax(1)

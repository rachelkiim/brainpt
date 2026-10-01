"""EEG complexity / entropy measures."""
import numpy as np
from scipy.signal import welch

def sample_entropy(x, m=2, r_ratio=0.2, max_n=3000):
    """Sample Entropy (Richman & Moorman 2000)"""
    x=np.asarray(x,dtype=float)
    if len(x)>max_n:  # downsample to limit cost
        step=len(x)//max_n
        x=x[::step]
    N=len(x)
    if N<m+2: return np.nan
    r=r_ratio*np.std(x)
    if r==0: return np.nan
    def count(mm):
        X=np.array([x[i:i+mm] for i in range(N-mm)])
        c=0
        for i in range(len(X)):
            d=np.max(np.abs(X-X[i]),axis=1)
            c+=np.sum(d<=r)-1
        return c
    B=count(m); A=count(m+1)
    return float(-np.log(A/B)) if (A>0 and B>0) else np.nan

def perm_entropy(x, order=3, delay=1):
    """Permutation Entropy (Bandt & Pompe 2002), normalized."""
    x=np.asarray(x,dtype=float); N=len(x)
    if N < order*delay+1: return np.nan
    from itertools import permutations
    perms=list(permutations(range(order)))
    pmap={p:i for i,p in enumerate(perms)}
    counts=np.zeros(len(perms))
    for i in range(N-(order-1)*delay):
        seg=x[i:i+order*delay:delay]
        counts[pmap[tuple(np.argsort(seg))]]+=1
    p=counts[counts>0]/counts.sum()
    return float(-np.sum(p*np.log(p))/np.log(len(perms)))

def spectral_entropy(x, fs, band=(1,40)):
    """Spectral Entropy: normalized Shannon entropy of the PSD."""
    f,P=welch(x,fs=fs,nperseg=int(fs*2),noverlap=int(fs))
    m=(f>=band[0])&(f<=band[1])
    P=P[m]
    if P.sum()<=0: return np.nan
    p=P/P.sum()
    p=p[p>0]
    return float(-np.sum(p*np.log(p))/np.log(len(p)))

def lziv(x, max_n=20000):
    """Lempel-Ziv complexity (median binarization, normalized)."""
    x=np.asarray(x,dtype=float)
    if len(x)>max_n:
        x=x[::max(1,len(x)//max_n)]
    s=''.join('1' if v>np.median(x) else '0' for v in x)
    n=len(s)
    if n<2: return np.nan
    # Kaspar-Schuster
    i=0; c=1; seen=set()
    while i < n:
        j=i+1
        while j<=n and s[i:j] in seen:
            j+=1
        seen.add(s[i:j])
        c+=1
        i=j
    b=n/np.log2(n)
    return float(c/b)

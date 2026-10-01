"""Cross-Frequency Coupling"""

import numpy as np
from scipy.signal import hilbert, butter, filtfilt, welch

def bp(x, fs, lo, hi, order=4):
    ny=fs/2
    b,a=butter(order,[max(lo/ny,0.01), min(hi/ny,0.99)],btype='band')
    return filtfilt(b,a,x)

def pac_mi(x, fs, phase_band, amp_band, n_bins=18):
    """
    Phase-Amplitude Coupling — Modulation Index (Tort et al. 2010)
    0 = no coupling; larger = stronger coupling.
    """
    xp=bp(x,fs,*phase_band); xa=bp(x,fs,*amp_band)
    ph=np.angle(hilbert(xp)); am=np.abs(hilbert(xa))
    edges=np.linspace(-np.pi,np.pi,n_bins+1)
    m=np.zeros(n_bins)
    for i in range(n_bins):
        sel=(ph>=edges[i])&(ph<edges[i+1])
        m[i]=am[sel].mean() if sel.any() else 0
    if m.sum()<=0: return np.nan
    p=m/m.sum(); p=p[p>0]
    H=-np.sum(p*np.log(p))
    return float((np.log(n_bins)-H)/np.log(n_bins))

def pac_amp_corr(x, fs, b1, b2):
    """Power-to-power coupling: correlation of the two band amplitude envelopes."""
    a1=np.abs(hilbert(bp(x,fs,*b1))); a2=np.abs(hilbert(bp(x,fs,*b2)))
    if np.std(a1)==0 or np.std(a2)==0: return np.nan
    return float(np.corrcoef(a1,a2)[0,1])

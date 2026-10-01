"""Additional measures: aperiodic (FOOOF), alpha peak, within-block trend, variability."""
import numpy as np, os, re, pickle, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'common', 'eeg'))
from wpli_analysis import load_marker
from eeg_pipeline import run_preprocessing, EEG_CHANNELS
from scipy.signal import welch
from fooof import FOOOF

def block_metrics(seg, fs):
    out={}
    # Channel-mean PSD -> FOOOF
    avg=seg.mean(axis=0)
    f,p=welch(avg,fs=fs,nperseg=int(fs*2),noverlap=int(fs))
    try:
        fm=FOOOF(verbose=False,max_n_peaks=6); fm.fit(f,p,(1,40))
        out['ap_offset']=fm.aperiodic_params_[0]; out['ap_exp']=fm.aperiodic_params_[1]
    except Exception:
        out['ap_offset']=np.nan; out['ap_exp']=np.nan
    # Alpha peak frequency (7-13 Hz)
    idx=(f>=7)&(f<=13)
    out['alpha_peak']=float(f[idx][np.argmax(p[idx])]) if idx.any() else np.nan
    # Within-block trend and variability (30 s windows)
    win=int(30*fs); n=seg.shape[1]//win
    if n>=3:
        for band,(lo,hi) in {'Theta':(4,8),'Alpha':(8,12),'Beta':(12,30)}.items():
            vals=[]
            for i in range(n):
                s=avg[i*win:(i+1)*win]
                ff,pp=welch(s,fs=fs,nperseg=int(fs*2),noverlap=int(fs))
                m=(ff>=lo)&(ff<=hi)
                vals.append(10*np.log10(pp[m].mean()))
            vals=np.array(vals)
            t=np.arange(n)
            out[f'slope_{band}']=float(np.polyfit(t,vals,1)[0])
            out[f'cv_{band}']=float(np.std(vals)/abs(np.mean(vals))) if np.mean(vals)!=0 else np.nan
    return out

def process(edf, mk, from_csv=False, trim_s=30, trim_e=20):
    if from_csv:
        from csv_loader import run_preprocessing_from_csv
        cleaned,orig_n=run_preprocessing_from_csv(edf)
    else:
        cleaned,orig_n=run_preprocessing(edf)
    rows=load_marker(mk); fs=cleaned.info['sfreq']; scale=cleaned.n_times/orig_n
    ev={}
    for lat,dur,typ in rows:
        s=max(0,round(lat*scale)); e=min(cleaned.n_times,round((lat+dur)*scale))
        ev[typ]=(s,e)
    if 'baseline' in ev:
        s,e=ev['baseline']; ev['baseline']=(s+int(trim_s*fs),e-int(trim_e*fs))
    data=cleaned.get_data(); out={}
    for cond,(s,e) in ev.items():
        if (e-s)/fs<15: continue
        for k,v in block_metrics(data[:,s:e],fs).items():
            out[(cond,k)]=v
    return out

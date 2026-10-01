"""
Aperiodic (1/f) change vs eye/muscle artifact (Study 1, visit 1).

Same preprocessing as eeg_pipeline, but keeps the time courses of the ICA components removed as
eye/muscle. FOOOF offset/exponent from the cleaned data are compared per condition with the
activity of those components, and the two are correlated across participants.
"""
import numpy as np
import mne
from scipy.signal import welch
from fooof import FOOOF
mne.set_log_level('WARNING')

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'common', 'eeg'))
from eeg_pipeline import EEG_CHANNELS, detect_bad_channels

def run_with_artifact_tracking(edf_path, marker_rows, baseline_trim_start=30, baseline_trim_end=20, seed=97):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.pick(EEG_CHANNELS)
    raw.set_channel_types({ch: 'eeg' for ch in EEG_CHANNELS})
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='warn')
    fs = raw.info['sfreq']
    orig_n_times = raw.n_times

    raw.filter(l_freq=1.0, h_freq=40.0, fir_design='firwin', verbose=False)
    bads = detect_bad_channels(raw)
    raw.info['bads'] = bads

    good_raw = raw.copy().drop_channels(bads) if bads else raw.copy()
    good_raw.set_eeg_reference('average', verbose=False)

    n_components = min(len(good_raw.ch_names) - 1, 10)
    ica = mne.preprocessing.ICA(n_components=n_components, method='infomax',
                                  fit_params=dict(extended=True), random_state=seed, max_iter=500)
    ica.fit(good_raw, verbose=False)

    from mne_icalabel import label_components
    nyquist = fs / 2.0
    wide_raw = good_raw.copy()
    wide_raw.filter(l_freq=1.0, h_freq=min(100.0, nyquist-1), fir_design='firwin', verbose=False)
    ic_labels = label_components(wide_raw, ica, method='iclabel')
    labels = ic_labels['labels']
    probs = ic_labels['y_pred_proba']
    exclude = [i for i, (l, p) in enumerate(zip(labels, probs))
               if l in ('eye blink', 'muscle artifact', 'line noise', 'channel noise') and p >= 0.80]
    eye_muscle_idx = [i for i, l in enumerate(labels) if l in ('eye blink', 'muscle artifact')]

    # Component time courses before removal
    sources = ica.get_sources(good_raw).get_data()  # (n_components, n_times)

    # Cleaned data
    ica.exclude = exclude
    cleaned = good_raw.copy()
    ica.apply(cleaned, verbose=False)

    if bads:
        bad_raw = raw.copy().pick(bads)
        bad_raw.set_montage(montage, on_missing='warn')
        cleaned.add_channels([bad_raw], force_update_info=True)
        cleaned.info['bads'] = bads
        cleaned.interpolate_bads(reset_bads=True, verbose=False)
    cleaned.reorder_channels(EEG_CHANNELS)

    return cleaned, sources, eye_muscle_idx, labels, orig_n_times, fs


def get_block_bounds(marker_rows, orig_n_times, clean_n_times, fs, baseline_trim_start=30, baseline_trim_end=20,
                      min_block_default=30):
    scale = clean_n_times / orig_n_times
    events = {}
    for lat, dur, typ in marker_rows:
        s = max(0, round(lat*scale))
        e = min(clean_n_times, round((lat+dur)*scale))
        events[typ] = (s, e)
    if 'baseline' in events:
        s, e = events['baseline']
        s2 = s + int(baseline_trim_start*fs)
        e2 = e - int(baseline_trim_end*fs)
        if e2 > s2:
            events['baseline'] = (s2, e2)
    return events


def compute_fooof_params(data_1d, fs, freq_range=(1, 40)):
    win = int(fs*2)
    freqs, psd = welch(data_1d, fs=fs, nperseg=min(win, len(data_1d)), noverlap=min(win,len(data_1d))//2)
    fm = FOOOF(verbose=False, max_n_peaks=6)
    fm.fit(freqs, psd, freq_range)
    return fm.aperiodic_params_[0], fm.aperiodic_params_[1]  # offset, exponent

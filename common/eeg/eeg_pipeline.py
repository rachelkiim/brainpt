"""
EEG preprocessing pipeline (Study 1 and Study 2).

Flow:
1. Load EDF (14 EEG channels)
2. Filter 1-40Hz
3. Bad channel detection (pyprep NoisyChannels: flatline/deviation/correlation)
4. Common average reference
5. ICA (extended infomax)
6. ICLabel -> remove Eye/Muscle/Line Noise/Channel Noise >= 0.80
7. Bad channel interpolation
8. Rebuild events from marker file (scaleFactor correction)
9. QC (minimum block length: default 30s, brainpt 15s)
10. Optional baseline trimming (start / end)
11. Band power (Welch, window=min(1s,block), 50% overlap) per band
12. dB = 10*log10(P_cond/P_baseline)
"""
import numpy as np
import mne
from scipy.signal import welch
mne.set_log_level('WARNING')

EEG_CHANNELS = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
BANDS = {'Delta': (1, 4), 'Theta': (4, 8), 'Alpha': (8, 12), 'Beta': (12, 30)}
MIN_BLOCK_SEC_DEFAULT = 30
MIN_BLOCK_SEC_BRAINPT = 15


def detect_bad_channels(raw, line_noise_sd=4):
    from pyprep import NoisyChannels
    nc = NoisyChannels(raw, random_state=97)
    nc.find_bad_by_nan_flat()
    nc.find_bad_by_deviation(deviation_threshold=line_noise_sd)
    try:
        nc.find_bad_by_correlation()
    except Exception as e:
        print(f"  (correlation check skipped: {e})")
    bads = list(set(nc.get_bads()))
    if len(bads) > len(raw.ch_names) // 2:
        print(f"  WARNING: {len(bads)}개 채널이 bad로 탐지됨 - 과탐지 의심, 상위 3개만 유지")
        bads = bads[:3]
    return bads


def run_preprocessing(edf_path, seed=97):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.pick(EEG_CHANNELS)
    raw.set_channel_types({ch: 'eeg' for ch in EEG_CHANNELS})
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='warn')

    orig_n_times = raw.n_times

    raw.filter(l_freq=1.0, h_freq=40.0, fir_design='firwin', verbose=False)

    bads = detect_bad_channels(raw)
    raw.info['bads'] = bads
    print(f"  Bad channels detected: {bads}")

    good_raw = raw.copy().drop_channels(bads) if bads else raw.copy()
    good_raw.set_eeg_reference('average', verbose=False)

    n_components = min(len(good_raw.ch_names) - 1, 10)
    ica = mne.preprocessing.ICA(n_components=n_components, method='infomax',
                                  fit_params=dict(extended=True), random_state=seed, max_iter=500)
    ica.fit(good_raw, verbose=False)

    try:
        from mne_icalabel import label_components
        nyquist = raw.info['sfreq'] / 2.0
        h_freq_wide = min(100.0, nyquist - 1.0)
        wide_raw = raw.copy().drop_channels(bads) if bads else raw.copy()
        wide_raw.filter(l_freq=1.0, h_freq=h_freq_wide, fir_design='firwin', verbose=False)
        wide_raw.set_eeg_reference('average', verbose=False)
        ic_labels = label_components(wide_raw, ica, method='iclabel')
        labels = ic_labels['labels']
        probs = ic_labels['y_pred_proba']
        exclude = [i for i, (l, p) in enumerate(zip(labels, probs))
                   if l in ('eye blink', 'muscle artifact', 'line noise', 'channel noise') and p >= 0.80]
        print(f"  ICLabel labels: {labels}")
        print(f"  Excluded components: {exclude}")
    except Exception as e:
        print(f"  ICLabel failed ({e}), skipping IC removal")
        exclude = []

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

    return cleaned, orig_n_times


def run_preprocessing_split(edf_paths, seed=97):
    cleaned_parts = []
    orig_lens = []
    for p in edf_paths:
        c, orig_n = run_preprocessing(p, seed=seed)
        cleaned_parts.append(c)
        orig_lens.append(orig_n)

    merged = cleaned_parts[0].copy()
    if len(cleaned_parts) > 1:
        merged.append(cleaned_parts[1:])

    clean_offsets = [0]
    for c in cleaned_parts[:-1]:
        clean_offsets.append(clean_offsets[-1] + c.n_times)

    return merged, orig_lens, clean_offsets


def local_band_power(data_1ch, fs, band):
    data_1ch = np.asarray(data_1ch).flatten()
    if len(data_1ch) < fs:
        return np.nan
    win = min(int(fs), len(data_1ch))
    noverlap = win // 2
    freqs, psd = welch(data_1ch, fs=fs, nperseg=win, noverlap=noverlap)
    idx = (freqs >= band[0]) & (freqs <= band[1])
    if not idx.any():
        return np.nan
    return psd[idx].mean()


def segment_and_compute_db_multi(cleaned_raw, per_file_orig_n, per_file_clean_offset,
                                   marker_rows_per_file, min_block_default=30, min_block_brainpt=15,
                                   baseline_trim_start_sec=0, baseline_trim_end_sec=0,
                                   split_blocks=None):
    """
    split_blocks: e.g. {'shortform': 3} also computes 'shortform_p1'..'shortform_p3'
    (the block split into thirds, in time order); the whole-block result is kept.
    """
    if split_blocks is None:
        split_blocks = {}
    fs = cleaned_raw.info['sfreq']
    data = cleaned_raw.get_data()
    ch_names = cleaned_raw.ch_names

    events = []
    for file_idx, marker_rows in enumerate(marker_rows_per_file):
        orig_n = per_file_orig_n[file_idx]
        if file_idx < len(per_file_clean_offset) - 1:
            file_clean_len = per_file_clean_offset[file_idx+1] - per_file_clean_offset[file_idx]
        else:
            file_clean_len = cleaned_raw.n_times - per_file_clean_offset[file_idx]
        scale_factor = file_clean_len / orig_n if orig_n > 0 else 1.0
        offset = per_file_clean_offset[file_idx]
        for lat, dur, typ in marker_rows:
            new_lat = offset + max(0, round(lat * scale_factor))
            new_dur = max(1, round(dur * scale_factor))
            if new_lat >= cleaned_raw.n_times:
                continue
            if new_lat + new_dur > cleaned_raw.n_times:
                new_dur = cleaned_raw.n_times - new_lat
            events.append({'type': typ, 'start': new_lat, 'end': new_lat + new_dur})

    if baseline_trim_start_sec > 0 or baseline_trim_end_sec > 0:
        trim_start_samp = int(round(baseline_trim_start_sec * fs))
        trim_end_samp = int(round(baseline_trim_end_sec * fs))
        for ev in events:
            if ev['type'] == 'baseline':
                orig_len_sec = (ev['end'] - ev['start']) / fs
                new_start = ev['start'] + trim_start_samp
                new_end = ev['end'] - trim_end_samp
                if new_end <= new_start:
                    print(f"  WARNING: baseline 트리밍 후 구간이 사라짐 (원래 {orig_len_sec:.1f}초) - 트리밍 생략")
                    continue
                print(f"  baseline 트리밍: {orig_len_sec:.1f}초 -> {(new_end-new_start)/fs:.1f}초 "
                      f"(앞 {baseline_trim_start_sec}s, 뒤 {baseline_trim_end_sec}s 제외)")
                ev['start'], ev['end'] = new_start, new_end

    valid_events = []
    for ev in events:
        remain_sec = (ev['end'] - ev['start']) / fs
        min_sec = min_block_brainpt if 'brainpt' in ev['type'] else min_block_default
        status = 'OK' if remain_sec >= min_sec else 'TOO_SHORT'
        print(f"  {ev['type']:12s} | remaining = {remain_sec:.2f} sec | {status}")
        if remain_sec >= min_sec:
            valid_events.append(ev)

    if not any(ev['type'] == 'baseline' for ev in valid_events):
        raise ValueError("baseline block missing or too short - dB normalization impossible")

    # Split blocks into equal parts (within-block time trend)
    split_events = []
    for ev in valid_events:
        if ev['type'] in split_blocks:
            n = split_blocks[ev['type']]
            total_len = ev['end'] - ev['start']
            part_len = total_len // n
            for i in range(n):
                p_start = ev['start'] + i * part_len
                p_end = ev['start'] + (i + 1) * part_len if i < n - 1 else ev['end']
                split_events.append({'type': f"{ev['type']}_p{i+1}", 'start': p_start, 'end': p_end})
                print(f"  {ev['type']}_p{i+1}   | {(p_end-p_start)/fs:.1f}초 구간 (등분)")
    valid_events = valid_events + split_events

    results = {}
    for ev in valid_events:
        seg = data[:, ev['start']:ev['end']]
        cond_result = {}
        for i, ch in enumerate(ch_names):
            band_result = {}
            for band_name, band_range in BANDS.items():
                band_result[band_name] = local_band_power(seg[i], fs, band_range)
            cond_result[ch] = band_result
        results[ev['type']] = cond_result

    baseline = results['baseline']
    db_results = {}
    for cond, ch_data in results.items():
        if cond == 'baseline':
            continue
        db_results[cond] = {}
        for ch in ch_names:
            db_results[cond][ch] = {}
            for band_name in BANDS:
                p_cond = ch_data[ch][band_name]
                p_base = baseline[ch][band_name]
                if np.isnan(p_cond) or np.isnan(p_base) or p_base == 0:
                    db_results[cond][ch][band_name] = np.nan
                else:
                    db_results[cond][ch][band_name] = 10 * np.log10(p_cond / p_base)

    return db_results, results


def segment_and_compute_db(cleaned_raw, orig_n_times, marker_rows, min_block_default=30, min_block_brainpt=15,
                             baseline_trim_start_sec=0, baseline_trim_end_sec=0, split_blocks=None):
    return segment_and_compute_db_multi(
        cleaned_raw,
        per_file_orig_n=[orig_n_times],
        per_file_clean_offset=[0],
        marker_rows_per_file=[marker_rows],
        min_block_default=min_block_default,
        min_block_brainpt=min_block_brainpt,
        baseline_trim_start_sec=baseline_trim_start_sec,
        baseline_trim_end_sec=baseline_trim_end_sec,
        split_blocks=split_blocks,
    )

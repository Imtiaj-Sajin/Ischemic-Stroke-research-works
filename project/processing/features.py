"""
features.py — per-epoch hand-crafted EEG features for classical ML staging.

extract_features(x, fs) : x [n, C, 3000] -> feats [n, C*F_per_ch], plus names.
Per channel: band powers (abs+rel), total power, spectral entropy, spectral edge,
mean freq, time-domain (std, ptp, rms, skew, kurtosis, zero-cross rate),
Hjorth (activity, mobility, complexity). All vectorised over epochs.
"""
import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis

BANDS = [("delta", 0.5, 4), ("theta", 4, 8), ("alpha", 8, 13),
         ("sigma", 11, 16), ("beta", 16, 30)]


def _hjorth(x):                       # x [n, C, T]
    dx = np.diff(x, axis=-1)
    ddx = np.diff(dx, axis=-1)
    v0 = x.var(axis=-1) + 1e-12
    v1 = dx.var(axis=-1) + 1e-12
    v2 = ddx.var(axis=-1) + 1e-12
    activity = v0
    mobility = np.sqrt(v1 / v0)
    complexity = np.sqrt(v2 / v1) / (mobility + 1e-12)
    return activity, mobility, complexity


def extract_features(x, fs=100):
    n, C, T = x.shape
    freqs, psd = welch(x, fs=fs, nperseg=min(256, T), noverlap=128, axis=-1)  # [n,C,F]
    total = psd.sum(axis=-1) + 1e-12                                           # [n,C]
    feats, names = [], []

    for bname, lo, hi in BANDS:
        m = (freqs >= lo) & (freqs < hi)
        bp = psd[:, :, m].sum(axis=-1)                # [n,C]
        feats.append(bp);            names += [f"{bname}_abs_c{c}" for c in range(C)]
        feats.append(bp / total);    names += [f"{bname}_rel_c{c}" for c in range(C)]

    feats.append(np.log(total));     names += [f"logtotal_c{c}" for c in range(C)]

    pn = psd / total[:, :, None]                       # normalised spectrum
    sent = -(pn * np.log(pn + 1e-12)).sum(axis=-1)     # spectral entropy
    feats.append(sent);              names += [f"spec_entropy_c{c}" for c in range(C)]

    cumpsd = np.cumsum(psd, axis=-1) / total[:, :, None]
    edge_idx = (cumpsd < 0.95).sum(axis=-1).clip(max=len(freqs) - 1)
    feats.append(freqs[edge_idx]);   names += [f"spec_edge_c{c}" for c in range(C)]
    meanf = (psd * freqs[None, None, :]).sum(-1) / total
    feats.append(meanf);             names += [f"mean_freq_c{c}" for c in range(C)]

    feats.append(x.std(axis=-1));    names += [f"std_c{c}" for c in range(C)]
    feats.append(np.ptp(x, axis=-1)); names += [f"ptp_c{c}" for c in range(C)]
    feats.append(np.sqrt((x ** 2).mean(-1))); names += [f"rms_c{c}" for c in range(C)]
    feats.append(skew(x, axis=-1));  names += [f"skew_c{c}" for c in range(C)]
    feats.append(kurtosis(x, axis=-1)); names += [f"kurt_c{c}" for c in range(C)]
    zc = (np.diff(np.sign(x), axis=-1) != 0).mean(axis=-1)
    feats.append(zc);                names += [f"zcr_c{c}" for c in range(C)]

    act, mob, cplx = _hjorth(x)
    feats.append(np.log(act));       names += [f"hjorth_act_c{c}" for c in range(C)]
    feats.append(mob);               names += [f"hjorth_mob_c{c}" for c in range(C)]
    feats.append(cplx);              names += [f"hjorth_cplx_c{c}" for c in range(C)]

    F = np.concatenate([f.reshape(n, -1) for f in feats], axis=1).astype(np.float32)
    return np.nan_to_num(F), names


if __name__ == "__main__":
    # Smoke test against a built subject: 23 features per channel, 161 for the
    # 7-channel montage. Run build_npz_full.py first if the corpus is absent.
    import os
    npz = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "processed7", "SN1.npz")
    if not os.path.exists(npz):
        raise SystemExit(f"no built corpus at {npz} -- run first: "
                         "python build_npz_full.py --raw <dataset dir>")
    x = np.load(npz, allow_pickle=True)["x"].astype(np.float32)
    F, names = extract_features(x)
    print("x:", x.shape, "-> feats:", F.shape,
          f"({len(names)} names, {F.shape[1] // x.shape[1]}/channel)")
    print("sample names:", names[:6], "...", names[-3:])

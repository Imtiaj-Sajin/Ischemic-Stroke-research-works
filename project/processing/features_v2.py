"""
features_v2.py — enhanced per-epoch features: adds EVENT-based sleep features that AASM
scorers actually use, on top of the base spectral/Hjorth/time-domain set (features.py).

For the 7-channel montage [C4:M1, C3:M2, O2:M1, O1:M2, E1:M2, E2:M2, EMG]:
  EEG (0-3): sleep-spindle density & amplitude (sigma 11-16 Hz envelope events),
             slow-wave peak-to-peak (delta 0.5-4 Hz)   -> N2 / N3 discriminators
  EOG (4-5): eye-movement energy (derivative power)     -> REM / Wake
  EMG (6)  : tonic level (log-RMS, high percentile)      -> REM atonia / Wake

extract_features_v2(x) -> [n, 188] is the per-epoch representation handed to the
modelling half; temporal_context(F, 3) -> [n, 1316] is the contextual vector the
classical prior consumes.
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import extract_features  # base 23 features/channel


def _bp(x, lo, hi, fs=100):
    sos = butter(4, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def event_features(x, fs=100):
    """x [n, 7, 3000] -> event features [n, F], names."""
    n = x.shape[0]; feats = []; names = []
    for c in range(4):                      # EEG channels
        xc = x[:, c, :]
        sig = _bp(xc, 11, 16, fs)           # spindle band
        env = np.abs(hilbert(sig, axis=-1))
        thr = np.percentile(env, 90, axis=-1, keepdims=True)
        above = env > thr
        onsets = (above[:, 1:] & ~above[:, :-1]).sum(-1)   # ~ spindle event count
        feats += [onsets.astype(np.float32), env.mean(-1), env.std(-1)]
        names += [f"spindle_dens_c{c}", f"spindle_amp_c{c}", f"spindle_var_c{c}"]
        sw = _bp(xc, 0.5, 4, fs)            # slow-wave band
        feats += [np.ptp(sw, axis=-1), np.abs(sw).mean(-1)]
        names += [f"sw_p2p_c{c}", f"sw_amp_c{c}"]
    for c in (4, 5):                        # EOG: eye-movement energy
        d = np.diff(x[:, c, :], axis=-1)
        feats += [np.log((d ** 2).mean(-1) + 1e-6), np.percentile(np.abs(d), 95, axis=-1)]
        names += [f"eog_mov_c{c}", f"eog_p95_c{c}"]
    emg = x[:, 6, :]                        # EMG tonic level (atonia)
    feats += [np.log(np.sqrt((emg ** 2).mean(-1)) + 1e-6),
              np.percentile(np.abs(emg), 90, axis=-1),
              np.log(np.var(np.diff(emg, axis=-1), axis=-1) + 1e-6)]
    names += ["emg_logrms", "emg_p90", "emg_diffvar"]
    F = np.stack(feats, axis=1).astype(np.float32)
    return np.nan_to_num(F), names


def extract_features_v2(x, fs=100):
    """Base features (features.py) + event features. x [n,7,3000] -> [n, F], names."""
    Fb, nb = extract_features(x, fs=fs)
    Fe, ne = event_features(x, fs=fs)
    return np.concatenate([Fb, Fe], axis=1), nb + ne


def temporal_context(F, k=3):
    """Concatenate each epoch's feature vector with its +/-k neighbours.

    AASM scoring is contextual by definition -- a REM period continues until
    specific evidence ends it, and N1 is defined relative to what precedes it --
    so an isolated epoch has thrown away the rules that decide its own label.
    Edges are padded by repeating the first/last epoch rather than by zeros, so
    the first and last epochs of a night stay in-distribution instead of being
    handed a feature vector no real epoch could produce.

    With the 188-dim vector and k=3 this is the 1316-dim input the classical
    prior consumes: 188 x (2*3 + 1).

    Parameters
    ----------
    F : ndarray [n_epochs, n_features]
        Per-epoch features, in recording order, for ONE subject. Never call this
        across concatenated subjects -- it would splice one patient's night onto
        another's and leak across cross-validation folds.
    k : int
        Neighbours on each side. Default 3 = three minutes of context.

    Returns
    -------
    ndarray [n_epochs, n_features * (2k + 1)]
    """
    F = np.asarray(F)
    if F.ndim != 2:
        raise ValueError(f"expected [n_epochs, n_features], got {F.shape}")
    idx = np.clip(np.arange(len(F))[:, None] + np.arange(-k, k + 1)[None, :],
                  0, len(F) - 1)
    return F[idx].reshape(len(F), -1)


if __name__ == "__main__":
    # Smoke test against a built subject: 161 base + 27 event = 188, then 1316
    # with +/-3 epochs of context. Run build_npz_full.py first if absent.
    npz = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "processed7", "SN1.npz")
    if not os.path.exists(npz):
        raise SystemExit(f"no built corpus at {npz} -- run first: "
                         "python build_npz_full.py --raw <dataset dir>")
    x = np.load(npz, allow_pickle=True)["x"].astype(np.float32)
    F, names = extract_features_v2(x)
    ev = [n for n in names if any(k in n for k in ("spindle", "sw_", "eog", "emg"))]
    print("x", x.shape, "-> features", F.shape, f"({len(names)} total, {len(ev)} event)")
    print("event features:", ev)
    print("with +/-3 context:", temporal_context(F, 3).shape)

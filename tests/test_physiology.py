"""
Physiological validity tests for the feature stage.

The contract tests prove the feature matrix has the right shape. These prove it
has the right meaning: that the spindle feature actually responds to spindles,
that the slow-wave feature responds to slow waves, that the EMG feature sees
atonia and the EOG feature sees eye movements.

That distinction matters because shape checks pass on garbage. A channel-order
bug, an inverted filter band or a wrong sampling rate all leave a
[n, 188] matrix that is numerically fine and physiologically meaningless -- and
the model downstream would still train, just worse, with nothing to point at.

Each test builds a synthetic epoch with one known property and asserts the
feature that is supposed to detect it does. No recordings needed.
    pytest -q
"""
import numpy as np
import pytest

from config import EPOCH_SAMPLES, SFREQ
from features import extract_features
from features_v2 import event_features

# Channel roles in the fixed montage [C4:M1, C3:M2, O2:M1, O1:M2, E1:M2, E2:M2, EMG]
EEG, EOG, EMG = 0, 4, 6
N_CHANNELS = 7

# Synthetic epoch indices, one per stage-like signal.
WAKE, N2, N3, REM = 0, 1, 2, 3
N_EPOCHS = 4


def col(names, name):
    """Column index of a named feature."""
    return names.index(name)


def sine(freq, amp, rng=None, noise=2.0):
    t = np.arange(EPOCH_SAMPLES) / SFREQ
    x = amp * np.sin(2 * np.pi * freq * t)
    if rng is not None:
        x = x + rng.normal(0, noise, EPOCH_SAMPLES)
    return x.astype(np.float32)


def spindle_train(rng, freq=13.0, amp=40.0, n_bursts=5, burst_s=0.7):
    """Sigma-band bursts on a noise floor -- the graphoelement that defines N2."""
    x = rng.normal(0, 5, EPOCH_SAMPLES)
    t = np.arange(EPOCH_SAMPLES) / SFREQ
    width = int(burst_s * SFREQ)
    for b in range(n_bursts):
        start = int((b + 0.5) * EPOCH_SAMPLES / n_bursts) - width // 2
        sl = slice(start, start + width)
        x[sl] += amp * np.sin(2 * np.pi * freq * t[sl])
    return x.astype(np.float32)


@pytest.fixture(scope="module")
def stages():
    """[4, 7, 3000] tensor: one synthetic epoch per stage-like signal."""
    rng = np.random.default_rng(7)
    x = rng.normal(0, 5, (N_EPOCHS, N_CHANNELS, EPOCH_SAMPLES)).astype(np.float32)

    # Wake: posterior alpha, high chin tone.
    for c in range(4):
        x[WAKE, c] = sine(10.0, 30.0, rng)
    x[WAKE, EMG] = rng.normal(0, 30, EPOCH_SAMPLES)

    # N2: sleep spindles, moderate chin tone.
    for c in range(4):
        x[N2, c] = spindle_train(rng)
    x[N2, EMG] = rng.normal(0, 8, EPOCH_SAMPLES)

    # N3: high-voltage slow waves.
    for c in range(4):
        x[N3, c] = sine(1.5, 80.0, rng)
    x[N3, EMG] = rng.normal(0, 6, EPOCH_SAMPLES)

    # REM: low-amplitude mixed EEG, rapid eye movements, atonia.
    for c in range(4):
        x[REM, c] = rng.normal(0, 8, EPOCH_SAMPLES)
    for c in (EOG, EOG + 1):
        x[REM, c] = sine(2.0, 200.0, rng)
    x[REM, EMG] = rng.normal(0, 2, EPOCH_SAMPLES)
    return x


@pytest.fixture(scope="module")
def base(stages):
    return extract_features(stages)


@pytest.fixture(scope="module")
def events(stages):
    return event_features(stages)


# ---------------------------------------------------------------------------
# Spectral features track the rhythm that is actually present
# ---------------------------------------------------------------------------

def test_alpha_dominates_the_wake_epoch(base):
    F, names = base
    alpha = F[:, col(names, f"alpha_rel_c{EEG}")]
    assert alpha[WAKE] > alpha[N3]
    assert alpha[WAKE] > alpha[REM]


def test_delta_dominates_the_slow_wave_epoch(base):
    F, names = base
    delta = F[:, col(names, f"delta_rel_c{EEG}")]
    assert delta[N3] > delta[WAKE]
    assert delta[N3] > delta[N2]


def test_sigma_dominates_the_spindle_epoch(base):
    F, names = base
    sigma = F[:, col(names, f"sigma_rel_c{EEG}")]
    assert sigma[N2] > sigma[N3]
    assert sigma[N2] > sigma[REM]


def test_relative_power_is_amplitude_invariant():
    # Absolute power moves with electrode impedance and skull anatomy between
    # patients; the band's share of total power does not. That invariance is
    # why both are kept, and it is what makes the feature transfer across
    # subjects at all.
    rng = np.random.default_rng(3)
    quiet = sine(10.0, 20.0, rng, noise=1.0)
    loud = quiet * 5.0
    x = np.stack([quiet, loud])[:, None, :]
    F, names = extract_features(x)
    rel = F[:, col(names, "alpha_rel_c0")]
    absolute = F[:, col(names, "alpha_abs_c0")]
    assert np.isclose(rel[0], rel[1], rtol=1e-3)   # unchanged by gain
    assert absolute[1] > absolute[0] * 10          # absolute follows gain


def test_spectral_entropy_is_lower_for_a_pure_rhythm_than_for_noise():
    rng = np.random.default_rng(4)
    pure = sine(10.0, 50.0, rng, noise=0.5)
    noise = rng.normal(0, 50, EPOCH_SAMPLES).astype(np.float32)
    x = np.stack([pure, noise])[:, None, :]
    F, names = extract_features(x)
    ent = F[:, col(names, "spec_entropy_c0")]
    assert ent[0] < ent[1]


def test_hjorth_mobility_rises_with_frequency():
    # Mobility is the ratio of the standard deviation of the first derivative to
    # that of the signal -- a proxy for mean frequency, so it must be monotone
    # in the frequency of a pure tone.
    rng = np.random.default_rng(5)
    x = np.stack([sine(f, 50.0, rng, noise=0.5) for f in (2, 10, 25)])[:, None, :]
    F, names = extract_features(x)
    mob = F[:, col(names, "hjorth_mob_c0")]
    assert mob[0] < mob[1] < mob[2]


def test_spectral_edge_rises_with_frequency():
    rng = np.random.default_rng(6)
    x = np.stack([sine(f, 50.0, rng, noise=0.5) for f in (3, 20)])[:, None, :]
    F, names = extract_features(x)
    edge = F[:, col(names, "spec_edge_c0")]
    assert edge[0] < edge[1]


# ---------------------------------------------------------------------------
# Event features track the graphoelements a human scorer looks for
# ---------------------------------------------------------------------------

def test_spindle_amplitude_responds_to_spindles(events):
    E, names = events
    amp = E[:, col(names, f"spindle_amp_c{EEG}")]
    assert amp[N2] > amp[N3]
    assert amp[N2] > amp[REM]


def test_spindle_variance_separates_bursts_from_steady_sigma():
    # A spindle is a burst, not continuous sigma. The envelope variance is what
    # distinguishes the two; the mean alone cannot.
    rng = np.random.default_rng(8)
    bursty = spindle_train(rng)
    steady = sine(13.0, 18.0, rng, noise=5.0)
    x = np.zeros((2, N_CHANNELS, EPOCH_SAMPLES), np.float32)
    x[0, EEG], x[1, EEG] = bursty, steady
    E, names = event_features(x)
    var = E[:, col(names, f"spindle_var_c{EEG}")]
    assert var[0] > var[1]


def test_slow_wave_amplitude_responds_to_slow_waves(events):
    E, names = events
    for feat in (f"sw_amp_c{EEG}", f"sw_p2p_c{EEG}"):
        v = E[:, col(names, feat)]
        assert v[N3] > v[N2], feat
        assert v[N3] > v[REM], feat


def test_slow_wave_feature_ignores_spindle_band_activity():
    # The 0.5-4 Hz band-pass must reject sigma outright, or N2 epochs with dense
    # spindles would start scoring as N3.
    rng = np.random.default_rng(9)
    x = np.zeros((2, N_CHANNELS, EPOCH_SAMPLES), np.float32)
    x[0, EEG] = sine(1.5, 80.0, rng)     # slow wave
    x[1, EEG] = sine(13.0, 80.0, rng)    # spindle band, same amplitude
    E, names = event_features(x)
    sw = E[:, col(names, f"sw_amp_c{EEG}")]
    assert sw[0] > 10 * sw[1]


def test_eog_feature_responds_to_eye_movements(events):
    E, names = events
    mov = E[:, col(names, f"eog_mov_c{EOG}")]
    assert mov[REM] > mov[N3]
    assert mov[REM] > mov[N2]


def test_emg_feature_sees_rem_atonia(events):
    # REM is the one stage where the EEG looks like light sleep but the chin EMG
    # is at its floor. That combination is what identifies it, so the EMG
    # feature must order Wake above REM.
    E, names = events
    tone = E[:, col(names, "emg_logrms")]
    assert tone[WAKE] > tone[N2] > tone[REM]


def test_emg_percentile_agrees_with_log_rms(events):
    E, names = events
    rms = E[:, col(names, "emg_logrms")]
    p90 = E[:, col(names, "emg_p90")]
    assert np.argmax(rms) == np.argmax(p90) == WAKE


def test_event_features_are_computed_per_eeg_channel_independently():
    # Only channel 1 carries spindles; only channel 1 may light up.
    rng = np.random.default_rng(10)
    x = rng.normal(0, 5, (1, N_CHANNELS, EPOCH_SAMPLES)).astype(np.float32)
    x[0, 1] = spindle_train(rng)
    E, names = event_features(x)
    amps = [E[0, col(names, f"spindle_amp_c{c}")] for c in range(4)]
    assert amps[1] > 3 * max(amps[0], amps[2], amps[3])

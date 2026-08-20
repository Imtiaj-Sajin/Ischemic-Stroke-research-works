"""
Contract tests for the processing stage.

These pin down the numbers the paper quotes -- 23 features per channel, 161 base,
27 event, 188 combined, 1316 with context, five AASM stages, a seven-channel
montage -- so that a change to any of them fails loudly instead of silently
invalidating Section 3.2 of the manuscript and every model trained downstream.

They run on synthetic signals: no raw recordings, no MNE, about a second.
    pytest -q
"""
import numpy as np
import pytest

from channel_mapping import (CHANNEL_MAPPING, LABEL_MAPPING, REF_RENAME,
                             REF_RENAME_EEG, TARGET_CHANNELS, harmonise_rename)
from config import EPOCH_SECONDS, EPOCH_SAMPLES, SFREQ
from features import BANDS, extract_features
from features_v2 import event_features, extract_features_v2, temporal_context

N_EPOCHS = 6
N_CHANNELS = 7
BASE_PER_CHANNEL = 23
N_BASE = BASE_PER_CHANNEL * N_CHANNELS        # 161
N_EVENT = 27
N_COMBINED = N_BASE + N_EVENT                 # 188
CONTEXT_K = 3
N_CONTEXT = N_COMBINED * (2 * CONTEXT_K + 1)  # 1316


@pytest.fixture
def x():
    """A [6, 7, 3000] epoch tensor of broadband noise, in microvolts."""
    rng = np.random.default_rng(0)
    return rng.normal(0, 20, (N_EPOCHS, N_CHANNELS, EPOCH_SAMPLES)).astype(np.float32)


# ---------------------------------------------------------------------------
# Epoch geometry
# ---------------------------------------------------------------------------

def test_epoch_geometry_is_30s_at_100hz():
    assert EPOCH_SECONDS == 30.0
    assert SFREQ == 100
    assert EPOCH_SAMPLES == 3000
    assert EPOCH_SAMPLES == int(EPOCH_SECONDS * SFREQ)


def test_sampling_rate_satisfies_nyquist_for_every_scored_band():
    # Everything AASM scoring depends on lies below 30 Hz, so 100 Hz leaves the
    # whole scored range comfortably inside the Nyquist limit.
    assert max(hi for _, _, hi in BANDS) * 2 < SFREQ


def test_bands_are_the_five_scored_rhythms():
    # The manuscript text rounds two of these (it writes alpha 8-12 and sigma
    # 12-16); the code is ground truth and the text is the thing to correct.
    assert BANDS == [("delta", 0.5, 4), ("theta", 4, 8), ("alpha", 8, 13),
                     ("sigma", 11, 16), ("beta", 16, 30)]


# ---------------------------------------------------------------------------
# Label map
# ---------------------------------------------------------------------------

def test_label_map_is_the_five_aasm_stages():
    assert LABEL_MAPPING == {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}


def test_label_values_are_contiguous_from_zero():
    # A gap here would silently break bincount-based class weights and any
    # confusion matrix indexed by stage id.
    assert sorted(LABEL_MAPPING.values()) == list(range(5))


def test_non_aasm_tokens_are_not_scoreable_stages():
    # A (artefact), Artefact and Movement all appear in the shipped workbooks.
    # None may map to a stage: a scorer marking an epoch unscorable is
    # information, and guessing a stage for it would inject label noise.
    for token in ("A", "Artefact", "Movement", "?"):
        assert token not in LABEL_MAPPING


# ---------------------------------------------------------------------------
# Montage harmonisation -- the 39 to 99 subject fix
# ---------------------------------------------------------------------------

def test_montage_is_four_eeg_two_eog_one_emg():
    assert len(TARGET_CHANNELS) == N_CHANNELS
    kinds = [CHANNEL_MAPPING[c] for c in TARGET_CHANNELS]
    assert kinds.count("eeg") == 4
    assert kinds.count("eog") == 2
    assert kinds.count("emg") == 1


def test_montage_excludes_frontal_derivations():
    # F3/F4 exist in only a minority of subjects; including them would either
    # shrink the cohort or force per-subject channel counts, which breaks the
    # fixed tensor shape subject-independent CV needs.
    assert not [c for c in TARGET_CHANNELS if c.startswith(("F3", "F4"))]


def test_a1a2_subject_is_recovered_onto_the_m1m2_montage():
    # The whole point of the step: a subject recorded under the earlobe
    # convention must end up with the same seven channel names as one recorded
    # under the mastoid convention. Without the rename it is dropped for missing
    # channels -- that is the 39-versus-99 row of the ablation table.
    a1a2 = ["C4:A1", "C3:A2", "O2:A1", "O1:A2", "EOG1:A2", "EOG2:A2", "Chin 1", "ECG 2"]
    rmap = harmonise_rename(a1a2, REF_RENAME)
    renamed = [rmap.get(c, c) for c in a1a2]
    assert set(TARGET_CHANNELS).issubset(renamed)


def test_m1m2_subject_is_left_untouched():
    assert harmonise_rename(TARGET_CHANNELS, REF_RENAME) == {}


def test_rename_never_collides_when_both_conventions_are_present():
    # Renaming an A1/A2 channel onto an already-present M1/M2 target would be a
    # duplicate-name collision, which MNE raises on.
    both = ["C4:A1", "C4:M1", "C3:A2"]
    rmap = harmonise_rename(both, REF_RENAME)
    assert "C4:A1" not in rmap           # target already present -- skip
    assert rmap["C3:A2"] == "C3:M2"      # target absent -- safe to rename


def test_rename_is_a_no_op_on_an_unrelated_montage():
    assert harmonise_rename(["ECG 2", "SPO2", "Snore"], REF_RENAME) == {}


def test_rename_sources_and_targets_are_known_channels():
    for src, dst in REF_RENAME.items():
        assert src in CHANNEL_MAPPING, f"{src} has no declared modality"
        assert dst in CHANNEL_MAPPING, f"{dst} has no declared modality"
        assert CHANNEL_MAPPING[src] == CHANNEL_MAPPING[dst], f"{src} -> {dst} changes modality"


def test_eeg_only_map_is_a_subset_of_the_full_map():
    # build_npz.py (4-channel) and build_npz_full.py (7-channel) must harmonise
    # EEG identically, or the two corpora disagree about what channel c is.
    assert REF_RENAME_EEG.items() <= REF_RENAME.items()
    assert all(CHANNEL_MAPPING[dst] == "eeg" for dst in REF_RENAME_EEG.values())


def test_full_map_adds_the_eog_and_emg_variants():
    assert set(REF_RENAME) - set(REF_RENAME_EEG) == {"EOG1:A2", "EOG2:A2",
                                                     "Chin 1", "Chin 2"}


# ---------------------------------------------------------------------------
# Feature dimensionality -- the 161 / 27 / 188 / 1316 the paper quotes
# ---------------------------------------------------------------------------

def test_base_features_are_23_per_channel(x):
    F, names = extract_features(x)
    assert F.shape == (N_EPOCHS, N_BASE)
    assert len(names) == N_BASE
    assert F.shape[1] // N_CHANNELS == BASE_PER_CHANNEL


def test_base_feature_count_scales_with_channel_count():
    rng = np.random.default_rng(1)
    for c in (1, 4, 7):
        F, names = extract_features(rng.normal(0, 20, (2, c, EPOCH_SAMPLES)))
        assert F.shape[1] == BASE_PER_CHANNEL * c == len(names)


def test_event_features_are_27(x):
    E, names = event_features(x)
    assert E.shape == (N_EPOCHS, N_EVENT)
    assert len(names) == N_EVENT
    # 4 EEG x 5 (spindle density/amp/var, slow-wave p2p/amp) + 2 EOG x 2 + EMG x 3
    assert sum(n.startswith(("spindle", "sw_")) for n in names) == 20
    assert sum(n.startswith("eog") for n in names) == 4
    assert sum(n.startswith("emg") for n in names) == 3


def test_combined_is_188_with_base_features_first(x):
    F, names = extract_features_v2(x)
    assert F.shape == (N_EPOCHS, N_COMBINED)
    assert len(names) == N_COMBINED
    # Feature-importance plots index by position, so the base block must stay
    # ahead of the event block.
    assert names[:N_BASE] == extract_features(x)[1]


def test_temporal_context_is_1316(x):
    F, _ = extract_features_v2(x)
    C = temporal_context(F, CONTEXT_K)
    assert C.shape == (N_EPOCHS, N_CONTEXT)
    assert N_CONTEXT == 1316


def test_feature_names_are_unique(x):
    _, names = extract_features_v2(x)
    assert len(set(names)) == len(names)


def test_every_feature_column_has_a_name(x):
    F, names = extract_features_v2(x)
    assert F.shape[1] == len(names)


def test_features_are_float32_even_from_the_float16_corpus(x):
    # The corpus is stored float16 to halve disk; everything downstream of the
    # cast must be float32, so no arithmetic happens in half precision.
    F, _ = extract_features_v2(x.astype(np.float16).astype(np.float32))
    assert F.dtype == np.float32


# ---------------------------------------------------------------------------
# Numerical robustness
# ---------------------------------------------------------------------------

def test_dead_channel_does_not_produce_nan(x):
    # A flat channel makes variance, skew and kurtosis undefined. The corpus
    # audit flags such channels, but the extractor must still return finite
    # numbers rather than poisoning the whole feature matrix.
    dead = x.copy()
    dead[:, 2, :] = 0.0
    F, _ = extract_features_v2(dead)
    assert np.isfinite(F).all()


@pytest.mark.filterwarnings("ignore:Precision loss occurred")
def test_saturated_channel_does_not_produce_inf(x):
    # scipy warns that skew/kurtosis are unreliable on a constant channel. That
    # is the point of the test -- the values must still come back finite.
    sat = x.copy()
    sat[:, 0, :] = 3000.0
    F, _ = extract_features_v2(sat)
    assert np.isfinite(F).all()


def test_channels_are_never_mixed(x):
    # Each base feature is a block of C columns, one per channel. Permuting the
    # input channels must permute those columns and nothing else -- if it does
    # not, channel c means a different electrode for different subjects, which
    # is exactly the failure the montage step exists to prevent.
    perm = [3, 1, 0, 2, 5, 4, 6]
    F0, _ = extract_features(x)
    F1, _ = extract_features(x[:, perm, :])
    for b in range(BASE_PER_CHANNEL):
        block0 = F0[:, b * N_CHANNELS:(b + 1) * N_CHANNELS]
        block1 = F1[:, b * N_CHANNELS:(b + 1) * N_CHANNELS]
        assert np.allclose(block1, block0[:, perm], atol=1e-5)


def test_extraction_is_deterministic(x):
    a, _ = extract_features_v2(x)
    b, _ = extract_features_v2(x)
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# temporal_context semantics
# ---------------------------------------------------------------------------

def _ramp(n=8, f=5):
    return np.arange(n * f, dtype=np.float32).reshape(n, f)


def test_context_centre_block_is_the_epoch_itself():
    F = _ramp()
    assert np.array_equal(temporal_context(F, 2)[:, 2 * 5:3 * 5], F)


def test_context_pads_edges_by_repetition_not_zeros():
    # A zero-padded first epoch would be a feature vector no real epoch could
    # produce; repeating epoch 0 keeps it in-distribution.
    F = _ramp()
    C = temporal_context(F, 3)
    assert np.array_equal(C[0, :5], F[0])        # missing left neighbour
    assert np.array_equal(C[-1, -5:], F[-1])     # missing right neighbour
    assert np.array_equal(C[0, :5], C[0, 5:10])


def test_context_block_order_is_oldest_to_newest():
    F, k, row = _ramp(), 3, 4
    C = temporal_context(F, k)
    for j, offset in enumerate(range(-k, k + 1)):
        expected = F[np.clip(row + offset, 0, len(F) - 1)]
        assert np.array_equal(C[row, j * 5:(j + 1) * 5], expected)


def test_context_k_zero_is_the_identity():
    F = _ramp()
    assert np.array_equal(temporal_context(F, 0), F)


def test_context_rejects_non_2d_input():
    with pytest.raises(ValueError):
        temporal_context(np.zeros((4, 7, 3000)), 3)


def test_context_handles_a_night_shorter_than_the_window():
    # Degenerate but must not crash: every neighbour clips to the only epoch.
    assert temporal_context(np.ones((1, N_COMBINED), np.float32), 3).shape == (1, N_CONTEXT)

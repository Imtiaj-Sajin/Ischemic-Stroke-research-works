CHANNEL_MAPPING = {'C4:M1': 'eeg',
                'C3:M2': 'eeg',
                'O2:M1': 'eeg',
                'O1:M2': 'eeg',
                'F4:A1': 'eeg',
                'F3:A2': 'eeg',
                'F4:M1': 'eeg',
                'F3:M2': 'eeg',
                'C4:A1': 'eeg',
                'C3:A2': 'eeg',
                'O2:A1': 'eeg',
                'O1:A2': 'eeg',
                'ECG 2': 'ecg',
                'E1:M2': 'eog',
                'E2:M2': 'eog',
                'EOG1:A2': 'eog',
                'EOG2:A2': 'eog',
                'EMG': 'emg',
                'EMG2': 'emg',
                'Chin 1': 'emg',
                'Chin 2': 'emg',
                'SPO2': 'misc',
                'PLMl': 'misc',
                'PLMr': 'misc',
                'Snore': 'misc',
                'Pressure Snore': 'misc',
                'Pulse': 'misc',
                'Pleth': 'misc',
                'Pos.': 'misc',
                'Move.': 'misc',
                'Light': 'misc',
                'Sum Effort': 'resp',
                'Sum RIPs': 'resp',
                'Abdomen': 'resp',
                'RIP Abdomen': 'resp',
                'Pressure Flow': 'misc',
                'Flow Th': 'temperature',      
                'Thorax': 'misc',
                'RIP Thorax': 'misc',
                'Battery': 'misc',
    }
LABEL_MAPPING = {  
    "W": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "R": 4,
}

# ---------------------------------------------------------------------------
# Montage harmonisation.
#
# The cohort was acquired over three years under two referencing conventions:
# some nights name the derivations against the earlobes (A1/A2), others against
# the mastoids (M1/M2). They are the same physical measurement -- adjacent scalp
# sites with near-identical potentials, which is why the AASM manual accepts
# either -- recorded under two different naming eras in one lab.
#
# A loader that simply asks for the seven channels it wants therefore drops
# every subject on the other convention, silently and without an exception.
# Harmonising the names is what takes the usable cohort from 39 to 99 subjects,
# and it is the single largest accuracy contribution in the paper's ablation.
#
# A true re-reference is not possible here: it would need the original
# common-reference recording, which the release does not ship. Renaming is the
# correct operation, not an approximation of one.
# ---------------------------------------------------------------------------

#: EEG derivations only -- used by the earlier 4-channel variant (build_npz.py).
REF_RENAME_EEG = {
    "C4:A1": "C4:M1",
    "C3:A2": "C3:M2",
    "O2:A1": "O2:M1",
    "O1:A2": "O1:M2",
}

#: Full 7-channel montage -- EEG plus the EOG and chin-EMG name variants
#: (build_npz_full.py). EOG1/EOG2 are the older names for E1/E2.
REF_RENAME = {
    **REF_RENAME_EEG,
    "EOG1:A2": "E1:M2",
    "EOG2:A2": "E2:M2",
    "Chin 1": "EMG",
    "Chin 2": "EMG2",
}

#: The fixed 7-channel montage, in the fixed order every subject is stored in:
#: 4 EEG + 2 EOG + 1 submental EMG. Frontal F3/F4 are deliberately excluded --
#: they exist in only a minority of subjects, so including them would either
#: shrink the cohort or force per-subject channel counts, which breaks the
#: fixed tensor shape that subject-independent cross-validation needs.
TARGET_CHANNELS = ["C4:M1", "C3:M2", "O2:M1", "O1:M2", "E1:M2", "E2:M2", "EMG"]


def harmonise_rename(ch_names, mapping=None):
    """Rename map to apply to `ch_names` to harmonise A1/A2 onto M1/M2.

    Returns only the entries that are safe to apply: the source name must be
    present and the target name must be absent. That guard matters -- a subject
    that somehow carries both conventions would otherwise hit a name collision
    on rename, and MNE raises rather than silently merging.

    Parameters
    ----------
    ch_names : sequence of str
        Channel names as read from the EDF.
    mapping : dict, optional
        Source-to-target names. Defaults to the full 7-channel :data:`REF_RENAME`.

    Returns
    -------
    dict
        Possibly empty; pass straight to ``epochs.rename_channels``.
    """
    mapping = REF_RENAME if mapping is None else mapping
    have = set(ch_names)
    return {src: dst for src, dst in mapping.items() if src in have and dst not in have}

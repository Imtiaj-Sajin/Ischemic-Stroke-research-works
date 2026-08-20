"""
build_npz_full.py — EDF + xlsx -> per-subject npz with the STANDARD sleep montage:
4 EEG (C4:M1, C3:M2, O2:M1, O1:M2) + 2 EOG (E1:M2, E2:M2) + 1 chin EMG.

We previously used EEG only, which handicaps REM (EOG eye-movements) and REM atonia
(EMG). This adds them, harmonising the A1/A2 reference and EOG/EMG name variants.
Output: data/processed7/SN<k>.npz  (x [n,7,3000] uV, y, channels, subject, sfreq).
"""
import os, sys, glob, argparse, warnings
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mne  # noqa
from staging_preprocess import StagingPreprocess  # noqa
from channel_mapping import (CHANNEL_MAPPING, LABEL_MAPPING, REF_RENAME,  # noqa
                             TARGET_CHANNELS, harmonise_rename)
from config import EPOCH_SECONDS, EPOCH_SAMPLES, SFREQ as TARGET_SFREQ  # noqa

warnings.simplefilter("ignore"); mne.set_log_level("ERROR")

TARGET = TARGET_CHANNELS          # montage and rename map live in channel_mapping.py,
                                  # so both drivers harmonise identically
WIN, SFREQ, SAMPLES = EPOCH_SECONDS, TARGET_SFREQ, EPOCH_SAMPLES  # from config.py
STAGE_NAMES = ["W", "N1", "N2", "N3", "R"]


def shortpath(p):
    """Path relative to the cwd for readable logs, absolute if that is not possible.

    Windows raises when the target is on a different drive from the working
    directory, which would otherwise crash the run at the very end -- after
    every subject has already been built.
    """
    try:
        return os.path.relpath(p)
    except ValueError:
        return os.path.abspath(p)


def process_subject(edf, ann, channels):
    ds = StagingPreprocess(edf, ann, CHANNEL_MAPPING, modality=["eeg", "eog", "emg"],
                           window_size=WIN, sfreq=SFREQ, preload=True)
    ep = StagingPreprocess.create_windows(ds.raw, ds.description, window_size=WIN,
                                          window_stride=WIN, label_mapping=LABEL_MAPPING,
                                          drop_last=True, drop_bad=True)
    rmap = harmonise_rename(ep.ch_names, REF_RENAME)
    if rmap:
        ep.rename_channels(rmap)
    missing = [c for c in channels if c not in ep.ch_names]
    if missing:
        raise RuntimeError(f"missing {missing}; have {ep.ch_names}")
    ep.pick(channels).reorder_channels(channels)
    x = ep.get_data(copy=False).astype(np.float32) * 1e6
    y = ep.metadata["target"].to_numpy().astype(np.int64)
    assert x.shape[1] == len(channels) and x.shape[2] == SAMPLES, x.shape
    assert x.shape[0] == y.shape[0]
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", action="append", default=None, help="raw dir(s); repeatable")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "data", "processed7"))
    ap.add_argument("--subjects", default="", help="comma list e.g. 1,2,3 (default all)")
    args = ap.parse_args()
    raws = args.raw or [os.path.join(HERE, "..", "data", "zenodo"),
                        os.path.join(HERE, "..", "data", "full100")]
    os.makedirs(args.out, exist_ok=True)
    edfs = {}
    for rd in raws:
        for p in glob.glob(os.path.join(rd, "SN*.edf")):
            sid = int(os.path.basename(p)[2:-4])
            edfs.setdefault(sid, p)  # first source wins (zenodo has SN2/13/17)
    if args.subjects:
        want = {int(s) for s in args.subjects.split(",") if s.strip()}
        edfs = {k: v for k, v in edfs.items() if k in want}
    ok = fail = skip = 0
    agg = np.zeros(5, dtype=np.int64)
    for sid in sorted(edfs):
        edf = edfs[sid]; ann = os.path.join(os.path.dirname(edf), f"SN{sid}.xlsx")
        if not os.path.exists(ann):
            continue
        outp = os.path.join(args.out, f"SN{sid}.npz")
        if os.path.exists(outp) and os.path.getsize(outp) > 1000:
            skip += 1; continue          # resume: skip already-processed subjects
        try:
            x, y = process_subject(edf, ann, TARGET)
        except Exception as e:
            print(f"[FAIL] SN{sid}: {type(e).__name__}: {e}"); fail += 1; continue
        # store signal as float16 to halve disk (spectral features unaffected); features re-cast to f32
        np.savez_compressed(outp, x=x.astype(np.float16), y=y,
                            channels=np.array(TARGET), subject=sid, sfreq=SFREQ)
        b = np.bincount(y, minlength=5); agg += b; ok += 1
        print(f"[ok] SN{sid}: {len(y):4d} ep  x{x.shape}  " +
              "  ".join(f"{n}:{c}" for n, c in zip(STAGE_NAMES, b)))
    if agg.sum():
        tot = int(agg.sum())
        print(f"\n=== {ok} subjects built, {tot} epochs ===")
        for n, c in zip(STAGE_NAMES, agg):
            print(f"  {n:3s} {c:6d}  ({100 * c / tot:4.1f}%)")
    print(f"\n=== done: ok={ok} skip={skip} fail={fail} -> {shortpath(args.out)} ===")


if __name__ == "__main__":
    main()

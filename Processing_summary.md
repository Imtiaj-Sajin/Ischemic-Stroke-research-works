# My part in one page — the processing stage

## The opening line

> "I owned the data pipeline. I turned the raw iSLEEPS release — 100 overnight sleep recordings
> as vendor files — into the labelled arrays the models train on, and I built the feature
> representation those models actually consume."

Your instinct ("I downloaded it, processed it, gave it to the team") is right, but *processing*
is the word that will get followed up. The five steps below are what it means.

---

## What processing actually was — 5 steps

**1. Get the data.** Two files per patient: `SNxx.edf` (the raw signals) and `SNxx.xlsx`
(the expert's sleep-stage annotations). Nothing is labelled or segmented yet.

**2. Decode the labels.** The hypnogram is buried in a 24-sheet vendor Excel export, with
8 junk header rows and a German column name. I wrote the parser and verified the rule holds on
all 40 workbooks — plus found three non-standard tokens (`A`, `Artefact`, `Movement`) that must
be thrown away rather than treated as a sleep stage.

**3. Fix the montage — this is the important one.** The hospital recorded over three years using
**two different electrode naming conventions**: some nights say `C4:A1` (earlobe reference),
others say `C4:M1` (mastoid reference). They are the same physical measurement with different
names. A loader that just asks for the seven channels it wants **silently drops every patient on
the other convention**. I harmonised the names, which is what took the usable cohort from
**39 patients to 99**.

**4. Cut into epochs.** Resample every recording to 100 Hz, attach the labels, cut into
non-overlapping 30-second windows = **3000 samples × 7 channels** (4 EEG + 2 EOG + 1 chin EMG),
scale to microvolts. One `.npz` file per patient.

**5. Build the features.** Turn each 30-second window into **188 numbers**: 161 spectral and
time-domain (band powers, spectral entropy, Hjorth), plus 27 that describe what a human scorer
actually looks for — sleep-spindle density, slow-wave amplitude, eye-movement energy, muscle tone.
Then each epoch is joined with its ±3 neighbours → **1316** numbers, because sleep staging rules
are contextual.

**And I audited it.** Found that **SN28 is a byte-identical duplicate of SN15** — that is why the
paper reports N=99, not 100. Leaving it in would put the same patient on both sides of a
cross-validation fold.

---

## The one fact that makes your part impressive

Table 3 of the paper (the ablation). Read the top block:

| Configuration | Patients | Accuracy | Macro-F1 |
|---|---|---|---|
| Classical prior, 4 EEG channels | 39 | 0.663 | 0.550 |
| Classical prior, 4 EEG channels | **99** | 0.727 | 0.644 |
| Classical prior, **7 channels** | 99 | 0.742 | 0.668 |
| Classical prior, 7 ch **+ event features** | 99 | **0.746** | **0.675** |
| + graph / state-space deep streams | 99 | 0.746 | 0.675 |

Every step up that column is a **processing** decision — more patients (my montage fix), more
channels (my montage design), better features (my feature engineering). Accuracy went
**0.663 → 0.746**. The deep-learning streams on top added **nothing**.

> "In our ablation, all of the accuracy came from the processing stage, not from model depth."

---

## If they ask "why was it complex / time-consuming?"

Three honest reasons:

1. **The annotation format is messy** — vendor Excel, not a clean label file.
2. **Two incompatible montage conventions** in one cohort, which silently destroys most of your
   data if you don't notice it.
3. **Scale** — 7.3 GB of recordings, where a half-downloaded file raises no error and would just
   quietly train on half a night. So I verify every file against the size its own EDF header
   declares.

---

## The handoff

What I gave my teammates:

- `SNxx.npz` per patient — `x` of shape `[n_epochs, 7, 3000]`, `y` of labels (W=0, N1=1, N2=2, N3=3, REM=4)
- `extract_features_v2(x)` → the 188-dim feature matrix
- `temporal_context(F, 3)` → the 1316-dim vector

They built everything downstream: the graph network, the state-space decoder, the HMM smoothing,
the evaluation. **Know the boundary — don't claim their part, and don't undersell yours.**

And the contract is enforced, not just documented — `pytest` at the repo root runs 62 tests in
about a second, on synthetic signals, with no recordings needed. They check the numbers above
(23 per channel, 161 + 27 = 188, 1316 with context), that the montage rename really does recover
an A1/A2 subject onto the M1/M2 montage, and that the features respond to the physiology they
claim to measure — spindles, slow waves, eye movements, atonia. If a teammate or an examiner
wants to know whether my half still holds, that is the one command to run.

---

## Numbers to remember

| | |
|---|---|
| Cohort | 100 stroke patients, **99 usable** (SN28 = SN15 duplicate) |
| Epoch | 30 seconds @ 100 Hz = **3000 samples** |
| Channels | **7** = 4 EEG (C4:M1, C3:M2, O2:M1, O1:M2) + 2 EOG + 1 chin EMG |
| Features | **188** per epoch (161 base + 27 event), **1316** with ±3 context |
| Total epochs | **95,305** |
| Stages | W 26% · N1 10% · N2 41% · N3 9% · REM 12% — heavily imbalanced |

---

*Full detail, the expected questions with answers, and the demo runbook are in
`Processing_viva_prep.docx`. The live demonstration is
`project/processing/iSLEEPS_preprocessing.ipynb`.*

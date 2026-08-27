# HAG-Net: Interpretable Sleep Staging in Subacute Ischemic Stroke

This repository contains the code, features, and manuscript for **HAG-Net**, a hybrid model for
automated sleep staging on polysomnography (PSG) from ischemic-stroke patients, together with the
data-processing pipeline that prepares the recordings.

> **HAG-Net: A Hemispheric Asymmetry-Guided Hybrid Network for Interpretable Sleep Staging and
> Lesion-Severity Biomarkers in Subacute Ischemic Stroke**
> Md Wahiduzzaman Suva, Esm-e Moula Chowdhury Abha, Md Imtiaj Alam Sajin
> Dept. of Computer Science and Engineering, American International University–Bangladesh.

---
Check the FINAL_SUBMISSION Folder regarding the paper, contribution, review, revision, and reproductibility.
and my (Imtiaj Sajin's) all contribuitions in the code is here: [project/processing](https://github.com/Imtiaj-Sajin/Ischemic-Stroke-research-works/tree/main/project/processing) ]
---
## The problem

Automated sleep staging works well on healthy sleepers, but it degrades sharply after focal brain
injury. A stroke lesion distorts exactly the micro-events a scorer relies on — sleep spindles,
slow waves, eye movements, muscle atonia — and it does so **asymmetrically**, so the two brain
hemispheres no longer look alike. Reliable staging matters clinically because sleep quality drives
neural recovery after stroke, and 70–80 % of stroke patients have sleep-disordered breathing.

The study uses **iSLEEPS**, the first public PSG corpus of subacute ischemic stroke: 99 usable
patients, **95,305** expertly scored 30-second epochs, five sleep stages (Wake, N1, N2, N3, REM).

## The central finding

On this cohort, **bigger models do not help**. Every convolutional, recurrent, transfer-learned,
and raw-signal deep network the paper tests *underperforms* classical hand-crafted features — the
signature of a **data-limited**, not capacity-limited, regime. 99 patients is simply too few for
end-to-end deep learning to win.

So HAG-Net is built to be *deep where deep helps and classical where classical helps*:

1. **Dual physiological feature representation** — 161 spectral/temporal + 27 event features
   (spindle density, slow-wave amplitude, ocular movement, EMG tone) per epoch.
2. **Classical prior stream** — an ensemble of gradient-boosting classifiers on those features;
   this is what actually carries the staging accuracy.
3. **Hemispheric-asymmetry graph stream** — graph attention over the electrode montage that pools
   *signed differences of homologous left/right derivations* to expose the lesion signal.
4. **Selective state-space (Mamba-style) temporal decoder** — models the overnight stage sequence.
5. **Residual-gated fusion + HMM decoding** — provably cannot score below the classical prior.

## Headline results

Under the dataset's **patient-exclusive ten-fold** protocol (all epochs of a patient are entirely
in train or test):

| Metric | HAG-Net | Published deep SOTA |
|---|---|---|
| Accuracy | 0.746 | 0.747 |
| Macro-F1 | up to 0.680 | 0.677 |
| Cohen's κ | 0.641 | 0.640 |

HAG-Net **matches** the deep state of the art on accuracy and **exceeds** it on the
minority-sensitive metrics (macro-F1, κ) that matter under heavy class imbalance, while dominating
every pure-deep baseline in subject-paired significance tests (*p* < 10⁻⁹).

Two results beyond staging accuracy:

- **Lesion-severity biomarker** — inter-hemispheric spindle asymmetry scales with stroke severity
  (Spearman ρ = 0.41, *p* = 0.006). The asymmetry stream doubles as a clinical instrument.
- **Calibrated uncertainty** — a split-conformal layer gives prediction sets with empirical
  coverage 0.900 at α = 0.1 and expected calibration error 0.038.

## Why processing is the story

The paper's ablation makes an unusual point: **essentially all the accuracy comes from the data
pipeline, not from model depth.**

| Configuration | Patients | Accuracy | Macro-F1 |
|---|---|---|---|
| Classical prior, 4 EEG channels | 39 | 0.663 | 0.550 |
| Classical prior, 4 EEG channels | 99 | 0.727 | 0.644 |
| Classical prior, 7 channels | 99 | 0.742 | 0.668 |
| Classical prior, 7 ch + event features | 99 | 0.746 | 0.675 |
| + graph / state-space deep streams | 99 | 0.746 | 0.675 |

Each gain is a preprocessing decision — recovering more patients, adding EOG/EMG channels,
engineering better features — and the deep streams on top add nothing to staging. That is why the
processing stage is documented here in its own right.

---

## Repository layout

```text
.
├── Sleep_Stage_Classification paper/   # the manuscript
│   ├── main.tex, references.bib
│   ├── compiled paper.pdf
│   └── figures/
├── project/
│   ├── processing/                     # the data pipeline (this repo's focus)
│   │   ├── iSLEEPS_preprocessing.ipynb #  → end-to-end pipeline, executed with outputs
│   │   ├── staging_preprocess.py       #  EDF read, annotation parse, resample, epoching
│   │   ├── channel_mapping.py          #  channel→modality map, AASM label map
│   │   ├── build_npz_full.py           #  driver: 7-channel montage → one .npz per subject
│   │   ├── build_npz.py                #  earlier EEG-only (4-channel) variant
│   │   ├── features.py                 #  161 base spectral/temporal features
│   │   ├── features_v2.py              #  27 physiological-event features (+ combined 188)
│   │   └── numpy_subjects.py           #  per-epoch file combiner (reference)
│   └── data/                           # built corpus — git-ignored, regenerated locally
├── tests/                               # contract, physiology and annotation tests (pytest)
├── dataset/                            # raw iSLEEPS recordings — git-ignored (see its README)
├── requirements.txt                     # pinned-minimum runtime + test dependencies
└── Processing_summary.md               # plain-language overview of the processing stage
```

## The processing pipeline

`project/processing/iSLEEPS_preprocessing.ipynb` is the executable pipeline — raw vendor files at
the top, model-ready tensors at the bottom, all outputs saved. It turns each patient's
`SNxx.edf` + `SNxx.xlsx` into:

- `SNxx.npz` — signals `x [n_epochs, 7, 3000]` (µV, 100 Hz, 30 s epochs) and labels
  `y [n_epochs]` ∈ {W=0, N1=1, N2=2, N3=3, REM=4};
- a **188-dimensional** per-epoch feature vector (161 base + 27 event), extended to **1316** with
  ±3 epochs of temporal context for the classical prior.

Key steps, and why each matters:

- **Annotation decoding** — the hypnogram is buried in a multi-sheet vendor Excel export; the
  parser is verified across all 40 open workbooks, and non-AASM tokens (`A`, `Artefact`,
  `Movement`) are dropped rather than mislabelled.
- **Montage harmonisation** — the cohort mixes two referencing conventions (A1/A2 and M1/M2).
  Renaming A1/A2 derivations to their M1/M2 equivalents is what takes the usable cohort from
  **39 to 99 patients**; without it a naive loader silently drops most of the dataset.
- **Epoching** — resample to 100 Hz, cut non-overlapping 30 s / 3000-sample windows over a fixed
  7-channel montage (4 EEG + 2 EOG + 1 chin EMG) so every patient has identical tensor shape.
- **Integrity audit** — duplicate-recording detection (SN28 is byte-identical to SN15, which is
  why N = 99, not 100), NaN/dead-channel/saturation checks, and EDF-header completeness
  verification for downloaded files.

## Data

**iSLEEPS** — Maiti, S., Sharma, S. K., Mythirayee, S., Rajendran, S. & Bapi, R. S.
*Polysomnography Dataset for Sleep Analysis in Ischemic Stroke Patients.* Scientific Data
**13**, 421 (2026). <https://doi.org/10.1038/s41597-026-06747-w>

- Subjects **SN1–SN40**: open on Zenodo — <https://doi.org/10.5281/zenodo.14873844>
- Subjects **SN41–SN100**: India Data Portal (free registration) —
  <https://india-data.org/dataset-details/0b801dfa-4e42-4ec6-9c56-c6892b907ed2>

The raw recordings are ~7.3 GB and are **not** committed to git. See
[`dataset/README.md`](dataset/README.md) for how to fetch and checksum-verify them. The
Sleep-EDF corpus (PhysioNet), used only for the healthy→stroke domain-gap experiment, is at
<https://physionet.org/content/sleep-edfx/>.

## Environment and hardware

The saved notebook was executed with **Python 3.11.3**, `numpy 2.2.6`, `scipy 1.15.3`,
`pandas 3.0.3`, `mne 1.12.1`, `scikit-learn 1.6.1`, `matplotlib 3.11.0` (see
`requirements.txt` for the minimum-supported versions; 3.10–3.12 are all fine). The
entire pipeline — EDF parsing, feature extraction, and the patient-exclusive
sanity-check classifier — runs on **CPU only**; no GPU is required or used. On a
single machine it processes one subject in a few seconds and the full open-40-subject
corpus in low single-digit minutes; the notebook prints per-stage timings as it runs.

## Getting started

```bash
# 1. dependencies
pip install -r requirements.txt

# 2. fetch the open subset into dataset/  (instructions in dataset/README.md)

# 3. build the per-subject corpus
python project/processing/build_npz_full.py --raw dataset --out project/data/processed7

# 4. or run the whole pipeline as a notebook, with narrative and figures
jupyter notebook project/processing/iSLEEPS_preprocessing.ipynb
```

The pipeline discovers whatever subjects are present, so it runs on the open 40 or the full 100.

## Verifying the pipeline

The processing stage makes claims the rest of the paper rests on: 23 features per
channel, 161 base plus 27 event equals 188, 1316 with context, five AASM stages, a
seven-channel montage, and a rename that recovers subjects recorded under the
earlobe convention. Those are checkable, so they are checked.

```bash
pytest        # 62 tests, about a second
```

The suite runs on synthetic signals -- no recordings, no download -- so it works on
a clean checkout:

| Suite | What it holds down |
|---|---|
| `tests/test_contract.py` | The dimensionality the manuscript quotes, the label map, the montage, and that permuting input channels only permutes feature columns -- so channel *c* cannot come to mean different electrodes for different subjects. |
| `tests/test_physiology.py` | That the features measure what they claim: the sigma feature responds to spindles and not to slow waves, the delta feature the reverse, the EOG feature to eye movements, the EMG feature to atonia. Shape checks pass on garbage; these do not. |
| `tests/test_annotations.py` | Vendor-workbook decoding: stage renaming, the 30-second onset grid, and non-AASM tokens being dropped rather than guessed. |

One test, `test_shipped_parser_drops_the_first_scored_epoch`, asserts a known
**defect** rather than correct behaviour. The annotation slice discards the first
scored epoch, which slides every remaining label one epoch earlier against the
signal. It is left in place deliberately: every number in the paper was produced
under this alignment, so re-aligning the corpus quietly would make our own
published results irreproducible. Read our scores as a lower bound. The fix is a
one-character change, it is the first item for the next revision, and when it
lands that test fails on purpose -- as the reminder to rebuild the corpus and
re-run the study.

## Results traceability

Every number this pipeline is responsible for maps to a checked location:

| Claim | Where it's produced | Where it's checked |
|---|---|---|
| 188 features/epoch (161 base + 27 event), 1316 with context | `features.py`, `features_v2.py`, notebook §8–9 | `tests/test_contract.py`, `tests/test_physiology.py` |
| 7-channel montage, A1/A2→M1/M2 harmonisation, 39→99 subjects | `build_npz_full.py`, notebook §3 | `tests/test_contract.py` |
| Annotation decoding, 30 s grid, non-AASM token dropping | `staging_preprocess.py`, notebook §2 | `tests/test_annotations.py` |
| SN28≡SN15 duplicate, N=99 not 100 | notebook §7 (integrity audit) | printed fingerprint check, re-run each execution |
| iSLEEPS stage distribution (Table 1, HAG-Net paper) | notebook §7 | recomputed live from the built corpus |

**MM-Net paper (Tables I, II, VII; Figures 4–10).** This author is also credited with
the cardiorespiratory feature set, the respiratory-baseline detectors, and these
result figures in that paper's contributions record. The feature/baseline source is
mirrored in [`project/processing/mmnet_extension/`](project/processing/mmnet_extension/)
with a provenance note; the figure-generation scripts and the feature/result cache
they read from are part of the main project repository
([`wshuv-o/isleeps-sleep-staging`](https://github.com/wshuv-o/isleeps-sleep-staging)),
consistent with that repository holding every training run (see its own README on
why one committer executed all GPU runs).

## Contributions

A three-person student project. The work split into a **data-processing** half
(pipeline, montage harmonisation, feature engineering, corpus auditing — everything in
`project/processing/`) and a **modelling** half (the HAG-Net graph and state-space streams,
fusion, HMM decoding, conformal calibration, biomarker analysis, and evaluation). This repository
is maintained by Md Imtiaj Alam Sajin, who owned the processing stage.

## Citation

```bibtex
@article{maiti2026isleeps,
  author  = {Maiti, Suvadeep and Sharma, Shivam Kumar and Mythirayee, S. and
             Rajendran, Srijithesh and Bapi, Raju S.},
  title   = {Polysomnography Dataset for Sleep Analysis in Ischemic Stroke Patients},
  journal = {Scientific Data}, volume = {13}, number = {1}, pages = {421}, year = {2026},
  doi     = {10.1038/s41597-026-06747-w}
}
```

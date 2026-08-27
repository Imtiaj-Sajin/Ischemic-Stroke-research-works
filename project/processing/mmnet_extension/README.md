# MM-Net cardiorespiratory extension — provenance note

The two scripts in this folder (`cardio_features.py`, `resp_baselines.py`) are the
cardiorespiratory-feature and respiratory-baseline-detector code credited to
Md Imtiaj Alam Sajin in the MM-Net paper's contributions record
(*"Trained and evaluated the respiratory reference detectors... implemented feature
extractors including the cardiorespiratory set"*). They are mirrored here, verbatim
from the commit that produced the paper's numbers, so this repository — not only the
lead author's — holds the source for the artifacts attributed to this author.

## What each script does

- **`cardio_features.py`** — extracts 14 physiologically meaningful summaries per
  epoch from the 7 cardiorespiratory channels (SpO2, pulse, ECG, airflow, thoracic/
  abdominal effort, thoraco-abdominal asynchrony), adds ±3-epoch temporal context,
  and retrains the gradient-boosting ensemble with EEG+cardio features against
  EEG-only, to test whether cardiorespiratory information helps staging.
- **`resp_baselines.py`** — benchmarks the respiratory-event detector against three
  reference points on identical folds: a no-training desaturation-depth rule,
  logistic regression, and gradient boosting on the same 14 cardio features — the
  baselines that calibrate MM-Net's respiratory AUC in Table VII of the paper.

## Why they cannot be re-run standalone from this repository

Both scripts import from the shared project's internal modules (`mmnet_repro`,
`datasets`) and read from a feature cache (`data/mm_features/SN*.npz`,
`data/multimodal/`) built by the main project's training pipeline, not by this
repository's preprocessing suite. That cache is regenerated on the lead author's
machine as part of the full MM-Net reproduction
(see `github.com/wshuv-o/isleeps-sleep-staging`, `1_MM_Net_reproduction.ipynb`,
which re-executes end to end in 82.4 minutes with saved outputs).

This split is the one already documented in the paper's contributions record: *"The
main project repository shows a single committer because every training run was
executed on the lead author's machine and GPU... the preprocessing suite ...
[was] developed in the co-authors' own repositories and are traceable there."*
Mirroring the source here makes the authorship and the code itself traceable to
this repository; reproducing the printed numbers end to end still requires the
shared feature cache in the main project repository.

Also mirrored here for the same reason: `make_result_figs.py`, `make_mm_figs.py`,
`regen_hypno_fig.py`, `fig_event_locked.py` — the scripts credited to this author
that produce Figures 4-10 and Tables I, II and VII. Same caveat applies: they read
from `results/*.csv` and `results/npz/*.npz` built by the shared pipeline, not from
this repository's own `project/data/`.

## Where their numbers land in the paper

| Script | Produces | In paper |
|---|---|---|
| `resp_baselines.py` | Desaturation-rule / logreg / gboost AUC and AP, 10-fold | Table VII (respiratory baselines) |
| `cardio_features.py` | EEG-only vs EEG+cardiorespiratory staging accuracy/macro-F1/κ | Ablation discussion, §VI-A |
| `make_result_figs.py` | Staging benchmark, confusion matrix, AHI, severity, training curve | Figs. 4, 6, 9; Table I, II |
| `make_mm_figs.py` | Per-class F1, modality-ablation grid, t-SNE embeddings | Figs. 7, 8, 10 |
| `regen_hypno_fig.py` | Whole-night hypnogram panel (patient SN90) | Fig. 5 |
| `fig_event_locked.py` | Per-event-type AUC (hypopnea/obstructive/central) | Fig. 8(b); Table VII |

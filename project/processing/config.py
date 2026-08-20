# Paths and modality used by the vendored official StagingPreprocess.
# Driver scripts (build_npz.py, build_npz_full.py) override these explicitly;
# the placeholders only need to exist so `from config import *` in
# staging_preprocess.py succeeds.
raw_file_path = 'data/zenodo'
output_data_path = 'data/processed'
modality = ['eeg']

# --- Epoch geometry -------------------------------------------------------
# 30 s is the AASM scoring unit, inherited from the paper-chart era where
# 30 seconds was one page at 10 mm/s. Every scorer is trained on it and every
# public hypnogram is scored on it, so deviating makes results incomparable.
#
# 100 Hz is the resampling target. Everything AASM scoring depends on lies
# below 30 Hz -- slow waves 0.5-4, spindles 11-16, beta to 30 -- so by Nyquist
# 100 Hz is comfortably sufficient, it is the de facto standard in the staging
# literature (Sleep-EDF is 100 Hz), and it cuts the corpus 2.56x versus the
# native 256 Hz of the release.
EPOCH_SECONDS = 30.0
SFREQ = 100
EPOCH_SAMPLES = int(EPOCH_SECONDS * SFREQ)   # 3000 samples per epoch per channel

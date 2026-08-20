"""
Annotation-decoding tests.

The hypnogram is not shipped as a label file. It is buried in a multi-sheet
vendor Excel export behind eight junk header rows, so decoding it is a parsing
job with real failure modes -- and every one of them is silent. A wrong slice
index shifts every label against the signal; an unmapped token turns an epoch
the scorer refused to score into a confident stage.

These tests build a miniature workbook with the same layout as the shipped ones
and assert the parser's behaviour on it, including the off-by-one described in
Processing_viva_prep.docx section 7. That test is deliberately written to assert
the CURRENT behaviour: it is a regression lock, not an endorsement. When the
slice is corrected for the next revision it will fail, and that failure is the
reminder that the whole corpus and every published number must be rebuilt.

Requires mne and openpyxl; skipped if either is absent.
    pytest -q
"""
import numpy as np
import pytest

pytest.importorskip("mne", reason="annotation parsing goes through mne.Annotations")
pytest.importorskip("openpyxl", reason="needed to write the test workbook")

import pandas as pd

from config import EPOCH_SECONDS
from staging_preprocess import StagingPreprocess

# Sheet rows 0-7 are vendor preamble; the first scored epoch sits on sheet row 8.
# pandas consumes sheet row 0 as the header, so the seven remaining preamble rows
# land at DataFrame indices 0-6 and the first scored epoch at index 7.
N_PREAMBLE_ROWS = 7


def write_workbook(path, stages, start="22:36:00"):
    """A minimal 'Sleep profile' sheet with the shipped layout: preamble + epochs."""
    preamble = [["Signal ID:", "Sleep profile"],
                ["Start Time:", start],
                ["Unterteilung:", "30 s"],
                ["Rate:", "1/30 Hz"],
                ["Events list:", "Wake, N1, N2, N3, REM, A"],
                ["Patient:", "SN1"],
                ["Recording:", "PSG"]]
    assert len(preamble) == N_PREAMBLE_ROWS
    rows = preamble + [[f"epoch_{i}", s] for i, s in enumerate(stages)]
    # The header row is written by to_excel from the column names.
    pd.DataFrame(rows, columns=["Signal ID:", "Sleep profile"]).to_excel(
        path, sheet_name="Sleep profile", index=False)
    return path


def parse(path):
    """Call the parser without needing an EDF -- it only uses window_size."""
    sp = object.__new__(StagingPreprocess)
    sp.window_size = EPOCH_SECONDS
    return sp.read_annotations(str(path))


@pytest.fixture
def workbook(tmp_path):
    def _make(stages):
        return write_workbook(tmp_path / "SN1.xlsx", stages)
    return _make


# ---------------------------------------------------------------------------
# Stage decoding
# ---------------------------------------------------------------------------

def test_the_five_stages_decode_to_their_aasm_tokens(workbook):
    # The workbook spells them Wake/N1/N2/N3/REM; LABEL_MAPPING keys on
    # W/N1/N2/N3/R, so Wake and REM are the two that must be renamed.
    ann = parse(workbook(["Wake", "N1", "N2", "N3", "REM"] * 2))
    # The first scored epoch is consumed by the slice (see the alignment test).
    assert list(ann.description) == ["N1", "N2", "N3", "R", "W", "N1", "N2", "N3", "R"]


def test_rem_is_renamed_to_r_to_match_the_label_map(workbook):
    # The workbook writes 'REM'; LABEL_MAPPING keys on 'R'. If this rename is
    # dropped, every REM epoch silently becomes unscoreable.
    ann = parse(workbook(["Wake"] + ["REM"] * 4))
    assert set(ann.description) == {"R"}


@pytest.mark.parametrize("token", ["Artefact", "Movement", "?", "", "N4", "Unscored"])
def test_non_aasm_tokens_become_bad_and_are_dropped_not_guessed(workbook, token):
    # MNE drops any epoch whose annotation starts with BAD_. A scorer marking an
    # epoch unscorable is information; assigning it a stage would inject label
    # noise into roughly 0.9% of epochs. ('A' is the one artefact token handled
    # separately -- see the next test.)
    ann = parse(workbook(["Wake", token, "N2"]))
    assert "BAD_?" in set(ann.description)
    assert token not in set(ann.description)


def test_artefact_token_a_is_kept_distinct_from_a_stage(workbook):
    # 'A' is passed through as 'A' rather than mapped to BAD_, but it is absent
    # from LABEL_MAPPING, so events_from_annotations never emits it as a stage.
    from channel_mapping import LABEL_MAPPING
    ann = parse(workbook(["Wake", "A", "A", "N2"]))
    assert "A" in set(ann.description)
    assert "A" not in LABEL_MAPPING


# ---------------------------------------------------------------------------
# Epoch grid
# ---------------------------------------------------------------------------

def test_onsets_are_a_regular_30_second_grid(workbook):
    ann = parse(workbook(["Wake"] * 12))
    onsets = np.asarray(ann.onset)
    assert np.allclose(np.diff(onsets), EPOCH_SECONDS)
    assert onsets[0] == 0.0


def test_every_epoch_has_the_full_duration(workbook):
    ann = parse(workbook(["N2"] * 6))
    assert np.allclose(ann.duration, EPOCH_SECONDS)


def test_onsets_are_regenerated_not_read_from_the_timestamp_column(workbook):
    # The parser rebuilds onsets as 30*i and ignores the timestamp column. That
    # is why a dropped row shifts everything after it rather than leaving a gap
    # -- the mechanism behind the alignment finding below.
    ann = parse(workbook(["Wake"] * 5))
    assert list(ann.onset) == [0.0, 30.0, 60.0, 90.0]


# ---------------------------------------------------------------------------
# The alignment finding -- regression lock on known, deliberate behaviour
# ---------------------------------------------------------------------------

def test_shipped_parser_drops_the_first_scored_epoch(workbook):
    """The [8:] slice discards one real epoch. Documented, deliberate, not fixed.

    pandas consumes sheet row 0 as the header, so DataFrame index 8 is sheet row
    9 -- but the first scored epoch is on sheet row 8. Because onsets are then
    regenerated as 30*i, every surviving label slides one epoch earlier against
    the signal.

    It is left in place because every number in the paper, and every model the
    modelling half trained, used this alignment. Silently re-aligning the corpus
    would make our own published results irreproducible. The fix is a one-
    character change and should lift every row of the leaderboard rather than
    overturn any conclusion.

    When that change lands, this test fails -- by design. Rebuild the corpus and
    re-run the study before updating it.
    """
    stages = ["Wake", "N1", "N2", "N3", "REM", "Wake"]
    ann = parse(workbook(stages))
    assert len(ann) == len(stages) - 1
    assert ann.description[0] == "N1", "first scored epoch ('Wake') should be the one dropped"


def test_the_drop_is_exactly_one_epoch_at_any_length(workbook):
    for n in (4, 10, 25):
        assert len(parse(workbook(["N2"] * n))) == n - 1

"""Unit tests for select_series: pure metadata logic, no DICOM files needed.

Rule: prefer Fluid_Sensitive=1; among remaining ties (or if none are fluid-sensitive),
take the lexicographically smallest SeriesInstanceUID. CSV row order must not matter.
"""

import pandas as pd
import pytest

from mri_core.rsna_knee_dataset import SERIES_COLUMNS, select_series

STUDY = "1.2.826.1"


def candidates(*rows):
    """rows: (SeriesInstanceUID, Fluid_Sensitive) tuples, all in one plane for one study."""
    return pd.DataFrame([(STUDY, uid, fluid, fluid, "Axial") for uid, fluid in rows], columns=SERIES_COLUMNS)


def test_fluid_sensitive_series_is_chosen_over_non_fluid():
    assert select_series(candidates(("A.non-fluid", 0), ("B.fluid", 1))) == "B.fluid"
    assert select_series(candidates(("Z.fluid", 1), ("A.non-fluid", 0))) == "Z.fluid"  # not just "first row"


@pytest.mark.parametrize("order", [(0, 1), (1, 0)], ids=["fluid-first", "non-fluid-first"])
def test_selection_is_stable_regardless_of_csv_row_order(order):
    rows = [("A.non-fluid", 0), ("B.fluid", 1)]
    ordered = [rows[i] for i in order]
    assert select_series(candidates(*ordered)) == "B.fluid"


def test_selection_is_stable_across_all_row_orderings_with_three_candidates():
    rows = [("C.non-fluid", 0), ("A.fluid", 1), ("B.fluid", 1)]
    import itertools
    results = {select_series(candidates(*perm)) for perm in itertools.permutations(rows)}
    assert results == {"A.fluid"}  # smallest SeriesInstanceUID among the fluid-sensitive candidates


def test_tie_break_among_fluid_sensitive_candidates_is_lexicographically_smallest_uid():
    assert select_series(candidates(("9.fluid", 1), ("1.fluid", 1), ("5.fluid", 1))) == "1.fluid"


def test_tie_break_among_non_fluid_candidates_is_lexicographically_smallest_uid():
    assert select_series(candidates(("9.plain", 0), ("1.plain", 0), ("5.plain", 0))) == "1.plain"


def test_fallback_to_non_fluid_when_no_fluid_sensitive_series_exists():
    assert select_series(candidates(("B.plain", 0), ("A.plain", 0))) == "A.plain"


def test_single_candidate_is_returned_regardless_of_fluid_flag():
    assert select_series(candidates(("only.fluid", 1))) == "only.fluid"
    assert select_series(candidates(("only.plain", 0))) == "only.plain"


def test_empty_candidates_raises():
    with pytest.raises(ValueError, match="no candidate series"):
        select_series(candidates().iloc[0:0])


def test_lexicographic_uid_comparison_is_string_not_numeric():
    # "10" < "2" as strings; this pins the documented tie-break rule against an accidental numeric sort.
    assert select_series(candidates(("10", 1), ("2", 1))) == "10"

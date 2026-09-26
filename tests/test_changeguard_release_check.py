"""Focused synthetic tests for the ChangeGuard release-check report/status logic.

Tests import and call the pure helper functions directly; no subprocess is
invoked here.  All assertions are about labelling and formatting only.
"""

import sys
from pathlib import Path

import pytest

# Make the scripts directory importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from changeguard_release_check import (
    RESULT_FAIL,
    RESULT_FULL_PASS,
    RESULT_TARGETED_PASS,
    build_report,
    choose_result_label,
    format_output_tail,
)


class TestChooseResultLabel:
    def test_quick_pass(self):
        assert choose_result_label("quick", 0) == RESULT_TARGETED_PASS

    def test_full_pass(self):
        assert choose_result_label("full", 0) == RESULT_FULL_PASS

    def test_quick_fail(self):
        assert choose_result_label("quick", 1) == RESULT_FAIL

    def test_full_fail(self):
        assert choose_result_label("full", 1) == RESULT_FAIL

    def test_nonzero_returncode_always_fails(self):
        for code in (1, 2, 127, -1):
            assert choose_result_label("full", code) == RESULT_FAIL
            assert choose_result_label("quick", code) == RESULT_FAIL

    def test_pass_label_never_contains_fail_word(self):
        for label in (RESULT_TARGETED_PASS, RESULT_FULL_PASS):
            assert "fail" not in label.lower()
            assert "failed" not in label.lower()

    def test_fail_label_never_contains_passed_word(self):
        assert "passed" not in RESULT_FAIL.lower()
        assert "ready" not in RESULT_FAIL.lower()

    def test_quick_pass_label_says_full_not_run(self):
        label = choose_result_label("quick", 0)
        assert "full regression suite not run" in label.lower()

    def test_full_pass_label_says_full_release(self):
        label = choose_result_label("full", 0)
        assert "full release check passed" in label.lower()


class TestFormatOutputTail:
    def test_short_output_returned_whole(self):
        raw = "line1\nline2\nline3"
        assert format_output_tail(raw, max_lines=30) == raw.strip()

    def test_long_output_truncated_to_max_lines(self):
        raw = "\n".join(f"line{i}" for i in range(100))
        result = format_output_tail(raw, max_lines=10)
        result_lines = result.splitlines()
        assert len(result_lines) <= 10
        assert "line99" in result

    def test_empty_string_returns_empty(self):
        assert format_output_tail("") == ""

    def test_leading_blank_lines_stripped(self):
        raw = "\n\n\nactual content"
        result = format_output_tail(raw)
        assert not result.startswith("\n")

    def test_preserves_last_lines_not_first(self):
        raw = "\n".join(f"L{i}" for i in range(50))
        result = format_output_tail(raw, max_lines=5)
        assert "L49" in result
        assert "L0" not in result


class TestBuildReport:
    def _quick_pass_report(self):
        return build_report("quick", 0, 1.23, "26 passed", "")

    def _full_fail_report(self):
        return build_report("full", 1, 4.56, "1 failed", "some error")

    def test_pass_report_contains_passed_label(self):
        report = self._quick_pass_report()
        assert RESULT_TARGETED_PASS in report

    def test_fail_report_contains_fail_label(self):
        report = self._full_fail_report()
        assert RESULT_FAIL in report

    def test_fail_report_does_not_contain_pass_label(self):
        report = self._full_fail_report()
        assert RESULT_TARGETED_PASS not in report
        assert RESULT_FULL_PASS not in report

    def test_report_contains_elapsed_time(self):
        report = build_report("quick", 0, 7.89, "", "")
        assert "7.89" in report

    def test_report_contains_exit_code(self):
        report = build_report("quick", 0, 1.0, "", "")
        assert "0" in report

    def test_report_contains_mode(self):
        report = build_report("full", 0, 1.0, "", "")
        assert "FULL" in report

    def test_stdout_tail_included_when_present(self):
        report = build_report("quick", 0, 1.0, "pytest output here", "")
        assert "pytest output here" in report

    def test_stderr_tail_included_when_present(self):
        report = build_report("quick", 1, 1.0, "", "some error text")
        assert "some error text" in report

    def test_empty_stdout_section_omitted(self):
        report = build_report("quick", 0, 1.0, "", "")
        assert "--- stdout (tail) ---" not in report

    def test_empty_stderr_section_omitted(self):
        report = build_report("quick", 0, 1.0, "", "")
        assert "--- stderr (tail) ---" not in report

    def test_nonzero_returncode_report_never_says_passed(self):
        for code in (1, 2, 127):
            report = build_report("quick", code, 1.0, "stuff", "")
            assert "PASSED" not in report

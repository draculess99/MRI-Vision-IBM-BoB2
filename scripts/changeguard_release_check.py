"""ChangeGuard release validation tool.

Cross-platform CLI for deterministic validation of ChangeGuard changes.

Usage
-----
Quick (targeted comparison tests only):
    .venv\\Scripts\\python.exe scripts\\changeguard_release_check.py --quick

Full (complete repository regression suite):
    .venv\\Scripts\\python.exe scripts\\changeguard_release_check.py --full

Exit codes: 0 = all selected tests passed, 1 = one or more tests failed.
"""

import argparse
import subprocess
import sys
import textwrap
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Maximum number of output lines shown in the summary (avoids wall-of-text).
_OUTPUT_TAIL_LINES = 30

RESULT_TARGETED_PASS = "TARGETED CHECK PASSED — full regression suite not run."
RESULT_FULL_PASS     = "FULL RELEASE CHECK PASSED."
RESULT_FAIL          = "FAILED — one or more tests did not pass."


# ---------------------------------------------------------------------------
# Pure helpers (importable; tested without subprocess)
# ---------------------------------------------------------------------------

def choose_result_label(mode: str, returncode: int) -> str:
    """Return the correct status label for a given mode and pytest returncode.

    Args:
        mode: "quick" or "full".
        returncode: The integer exit code returned by pytest.

    Returns:
        One of the three RESULT_* constants.
    """
    if returncode != 0:
        return RESULT_FAIL
    if mode == "quick":
        return RESULT_TARGETED_PASS
    return RESULT_FULL_PASS


def format_output_tail(raw: str, max_lines: int = _OUTPUT_TAIL_LINES) -> str:
    """Return the last *max_lines* lines of *raw*, stripped of leading blank lines."""
    lines = raw.splitlines()
    tail = lines[-max_lines:] if len(lines) > max_lines else lines
    return "\n".join(tail).strip()


def build_report(
    mode: str,
    returncode: int,
    elapsed: float,
    stdout_tail: str,
    stderr_tail: str,
) -> str:
    """Assemble the human-readable report string.

    This function is pure (no I/O) so it can be exercised directly in tests.
    """
    label = choose_result_label(mode, returncode)
    header = "=" * 60
    lines = [
        header,
        f"ChangeGuard release check — mode: {mode.upper()}",
        header,
        f"Status  : {label}",
        f"Elapsed : {elapsed:.2f}s",
        f"Exit    : {returncode}",
    ]
    if stdout_tail:
        lines += ["", "--- stdout (tail) ---", stdout_tail]
    if stderr_tail:
        lines += ["", "--- stderr (tail) ---", stderr_tail]
    lines += [header]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _pytest_command(mode: str) -> list:
    """Build the pytest argv list for the given mode."""
    base = [sys.executable, "-B", "-m", "pytest", "-p", "no:cacheprovider"]
    if mode == "quick":
        return base + ["-v", "tests/test_comparison.py"]
    # full
    return base + ["-q"]


def run_check(mode: str) -> int:
    """Execute the selected pytest command and print a report.

    Returns:
        The pytest process exit code (0 = pass, nonzero = fail).
    """
    cmd = _pytest_command(mode)
    start = time.monotonic()
    result = subprocess.run(
        cmd,
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        shell=False,      # explicit: never use shell=True
    )
    elapsed = time.monotonic() - start

    stdout_tail = format_output_tail(result.stdout)
    stderr_tail = format_output_tail(result.stderr)
    report = build_report(mode, result.returncode, elapsed, stdout_tail, stderr_tail)
    print(report)
    return result.returncode


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(__doc__ or ""),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--quick",
        action="store_true",
        help="Run targeted comparison tests only (tests/test_comparison.py).",
    )
    group.add_argument(
        "--full",
        action="store_true",
        help="Run the full repository pytest suite.",
    )
    args = parser.parse_args(argv)
    mode = "quick" if args.quick else "full"
    return run_check(mode)


if __name__ == "__main__":
    sys.exit(main())

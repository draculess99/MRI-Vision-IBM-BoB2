# ChangeGuard Release Validation Workflow

## The problem

ChangeGuard changes — threshold adjustments, new comparison features, UI wording — are
high-risk precisely because they affect what a clinician sees. Without a structured
release check, validation is manual and inconsistent: a developer may run the full suite
once, forget to run it at all, or conflate a passing smoke test with a green-light for
release.

`scripts/changeguard_release_check.py` gives every developer one command that produces
an honest, auditable result with no ambiguity about what was tested.

---

## The two commands

### Targeted check — fastest feedback

```powershell
.\.venv\Scripts\python.exe scripts\changeguard_release_check.py --quick
```

Runs `tests/test_comparison.py` only — the focused synthetic suite that covers every
comparison rule, threshold boundary, INVALID guard, delta arithmetic, and reason-string
wording.

**Output on success:**

```
============================================================
ChangeGuard release check — mode: QUICK
============================================================
Status  : TARGETED CHECK PASSED — full regression suite not run.
Elapsed : 0.52s
Exit    : 0
...
============================================================
```

This label is honest: it tells you the targeted tests passed **and** that you have not
yet verified the rest of the repository.

---

### Full regression check — required before commit/push

```powershell
.\.venv\Scripts\python.exe scripts\changeguard_release_check.py --full
```

Runs the complete pytest suite (`pytest -q -p no:cacheprovider`), which covers the
image-processing pipeline, DICOM loading, manifest/label validation, baseline features,
CNN inference wiring, Streamlit UI layout, and all comparison tests.

**Output on success:**

```
============================================================
ChangeGuard release check — mode: FULL
============================================================
Status  : FULL RELEASE CHECK PASSED.
Elapsed : 68.10s
Exit    : 0
...
============================================================
```

**Failure is never labeled as ready or passed.** Any non-zero pytest exit code produces:

```
Status  : FAILED — one or more tests did not pass.
```

regardless of mode.

---

## Developer workflow

```
┌─────────────────────────────────────────────────────────────┐
│  1. Scoped change                                           │
│     Edit only the allowed files (e.g. mri_core/comparison, │
│     app.py, or scripts/changeguard_release_check.py).       │
├─────────────────────────────────────────────────────────────┤
│  2. Focused test                                            │
│     Run --quick.  Fix any failures before continuing.       │
│     Do not proceed if FAILED appears.                       │
├─────────────────────────────────────────────────────────────┤
│  3. Full regression check                                   │
│     Run --full.  Confirm FULL RELEASE CHECK PASSED.         │
│     Zero tolerance for new failures or regressions.         │
├─────────────────────────────────────────────────────────────┤
│  4. Visual Streamlit verification                           │
│     Run:                                                    │
│       .\.venv\Scripts\python.exe -m streamlit run app.py   │
│     Load two RSNA studies, click Process Image, open the   │
│     ChangeGuard expander. Verify card colours, reason text, │
│     overlay images, and delta table render correctly.       │
├─────────────────────────────────────────────────────────────┤
│  5. Commit and push                                         │
│     Only after steps 2–4 are all green.                     │
└─────────────────────────────────────────────────────────────┘
```

---

## What this tool does not do

- It does not modify comparison thresholds, clinical logic, or decision rules.
- It does not make medical claims or clinical judgements.
- It does not call any network service, LLM, or external API.
- It does not install packages or write files outside the process.

The deterministic comparison logic in [`mri_core/comparison.py`](../mri_core/comparison.py)
and the quality-gate logic in [`mri_core/decision.py`](../mri_core/decision.py) are
**read-only** from the perspective of this tool. Running this script cannot change what
thresholds fire or what reason strings are produced.

---

## Exit codes

| Exit code | Meaning |
|---|---|
| `0` | All selected tests passed. |
| `1` | One or more tests failed; do not treat as ready for release. |

---

## Files

| File | Role |
|---|---|
| [`scripts/changeguard_release_check.py`](../scripts/changeguard_release_check.py) | CLI entry point; pure helpers `choose_result_label`, `build_report`, `format_output_tail` |
| [`tests/test_changeguard_release_check.py`](../tests/test_changeguard_release_check.py) | Focused synthetic tests for the report/status logic (no subprocess) |
| [`tests/test_comparison.py`](../tests/test_comparison.py) | The comparison-logic suite exercised by `--quick` |

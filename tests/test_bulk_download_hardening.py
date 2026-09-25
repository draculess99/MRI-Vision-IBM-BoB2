"""Tests for the bulk downloader's error reporting, retry classification, and circuit breaker.

Every Kaggle call is mocked (subprocess.run or download_file_with_retry); nothing touches the network,
data/rsna-knee, or any real cache. All files are written under pytest's tmp_path.
"""

import subprocess
import threading
import time
from pathlib import Path

import pytest

import scripts.bulk_download_labeled_studies as bd

REAL_SLEEP = time.sleep
VALID_DICOM = b"\0" * 128 + b"DICM" + b"\0" * 64
STUDY = "1.2.826.0.1.3680043.8.498.1000"
SERIES = "1.2.826.0.1.3680043.8.498.1000.1"


class FakeKaggle:
    """Stand-in for subprocess.run. Each outcome is 'ok', 'not-dicom', 'empty', or (returncode, stderr).
    The last outcome repeats once the list is exhausted."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def __call__(self, cmd, **kwargs):
        assert cmd[0] == "kaggle"
        self.calls += 1
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        staging = Path(cmd[cmd.index("-p") + 1])
        if outcome == "ok":
            (staging / "1.dcm").write_bytes(VALID_DICOM)
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if outcome == "not-dicom":
            (staging / "1.dcm").write_bytes(b"x" * 300)
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if outcome == "empty":
            return subprocess.CompletedProcess(cmd, 0, "", "")
        returncode, stderr = outcome
        return subprocess.CompletedProcess(cmd, returncode, "", stderr)


@pytest.fixture
def kaggle(monkeypatch):
    """Install a FakeKaggle and record (instead of performing) retry sleeps."""
    sleeps = []

    def install(*outcomes):
        fake = FakeKaggle(*outcomes)
        monkeypatch.setattr(bd.subprocess, "run", fake)
        return fake

    monkeypatch.setattr(bd.time, "sleep", lambda seconds: sleeps.append(seconds))
    install.sleeps = sleeps
    return install


def download(tmp_path, name="1.dcm"):
    return bd.download_file_with_retry(f"train_series/{STUDY}/{SERIES}/{name}", tmp_path / STUDY / SERIES / name)


# ------------------------------------------------------------------------------ 1-5: retry classification

@pytest.mark.parametrize("message", [
    "429 Client Error: Too Many Requests for url",
    "Too Many Requests",
    "500 Server Error: Internal Server Error",
    "504 Server Error: Gateway Time-out",
    "503 Server Error: Service Unavailable",
    "502 Bad Gateway",
    "Read timed out.",
    "Connection aborted.",
    "Service temporarily unavailable",
], ids=["429", "too-many-requests-text", "500", "504", "503", "502", "timed-out", "connection", "temporarily"])
def test_transient_errors_are_retried_then_succeed(kaggle, tmp_path, message):
    fake = kaggle((1, message), "ok")
    success, failure = download(tmp_path)
    assert success is True and failure is None
    assert fake.calls == 2
    assert kaggle.sleeps == [bd.RETRY_BACKOFF[0]]
    assert (tmp_path / STUDY / SERIES / "1.dcm").read_bytes() == VALID_DICOM


@pytest.mark.parametrize("message,category", [
    ("401 Client Error: Unauthorized", bd.CATEGORY_AUTH),
    ("403 Client Error: Forbidden", bd.CATEGORY_FORBIDDEN),
    ("404 Client Error: Not Found", bd.CATEGORY_NOT_FOUND),
])
def test_permanent_errors_are_not_retried(kaggle, tmp_path, message, category):
    fake = kaggle((1, message))
    success, failure = download(tmp_path)
    assert success is False
    assert fake.calls == 1 and kaggle.sleeps == []
    assert failure.category == category and failure.attempts == 1


def test_persistent_transient_error_stays_bounded(kaggle, tmp_path):
    fake = kaggle((1, "429 Client Error: Too Many Requests"))
    success, failure = download(tmp_path)
    assert success is False
    assert fake.calls == bd.MAX_RETRIES
    assert failure.attempts == bd.MAX_RETRIES
    assert kaggle.sleeps == bd.RETRY_BACKOFF[:bd.MAX_RETRIES - 1]
    assert failure.category == bd.CATEGORY_RATE_LIMIT


def test_status_numbers_inside_paths_and_uids_are_not_mistaken_for_http_codes(kaggle, tmp_path):
    for message in ["Failed for train_series/1.2.3/4.5.6/500.dcm",
                    "bad study 1.2.826.0.1.3680043.8.498.4291234567",
                    "kaggle: command not found"]:
        assert bd.classify_error(message) == bd.CATEGORY_UNKNOWN, message
        assert not bd.is_transient_error(message)
    fake = kaggle((1, "Failed for train_series/1.2.3/4.5.6/500.dcm"))
    download(tmp_path)
    assert fake.calls == 1


@pytest.mark.parametrize("message,category", [
    ("429 Too Many Requests", bd.CATEGORY_RATE_LIMIT),
    ("401 Unauthorized", bd.CATEGORY_AUTH),
    ("403 Forbidden", bd.CATEGORY_FORBIDDEN),
    ("503 Service Unavailable", bd.CATEGORY_SERVER_ERROR),
    ("Timeout after retries", bd.CATEGORY_TIMEOUT),
    ("Connection reset by peer", bd.CATEGORY_CONNECTION),
    ("Downloaded file is not valid DICOM", bd.CATEGORY_DICOM),
    ("No .dcm file in download", bd.CATEGORY_DICOM),
    ("something odd happened", bd.CATEGORY_UNKNOWN),
    ("", bd.CATEGORY_UNKNOWN),
])
def test_error_categories(message, category):
    assert bd.classify_error(message) == category


# ------------------------------------------------------------------------------ 6: capture and sanitizing

def test_failure_retains_returncode_path_attempts_and_message(kaggle, tmp_path):
    kaggle((2, "  something\n  broke  "))
    success, failure = download(tmp_path, "7.dcm")
    assert success is False
    assert failure.returncode == 2
    assert failure.file_path == f"train_series/{STUDY}/{SERIES}/7.dcm"
    assert failure.attempts == 1
    assert failure.message == "something broke"
    assert failure.category == bd.CATEGORY_UNKNOWN


def test_timeout_failure_has_no_returncode(kaggle, tmp_path, monkeypatch):
    def hang(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 60)

    monkeypatch.setattr(bd.subprocess, "run", hang)
    success, failure = download(tmp_path)
    assert success is False and failure.returncode is None
    assert failure.attempts == bd.MAX_RETRIES and failure.category == bd.CATEGORY_TIMEOUT


def test_error_text_is_sanitized_and_truncated(kaggle, tmp_path, monkeypatch):
    monkeypatch.setenv("KAGGLE_KEY", "supersecretkeyvalue123")
    home = str(Path.home())
    noisy = (f"bad credentials supersecretkeyvalue123 Authorization: Bearer abc.def-ghi token=XYZ789 "
             f"file {home}\\.kaggle\\credentials.json opaque {'A' * 60} " + "padding " * 100)
    kaggle((1, noisy))
    success, failure = download(tmp_path)
    assert "supersecretkeyvalue123" not in failure.message
    assert "abc.def-ghi" not in failure.message
    assert "XYZ789" not in failure.message
    assert "A" * 40 not in failure.message
    assert home not in failure.message
    assert "<redacted>" in failure.message
    assert len(failure.message) <= bd.ERROR_TEXT_LIMIT


@pytest.mark.parametrize("text,secret", [
    ("auth failed key 0123456789abcdef0123456789abcdef end", "0123456789abcdef0123456789abcdef"),
    ("bad token KGAT_0123456789abcdef0123456789abcdef end", "KGAT_0123456789abcdef0123456789abcdef"),
    ("GET https://x/api?k=KGAT_0123456789abcdef0123456789abcdef failed", "KGAT_0123456789abcdef0123456789abcdef"),
    ('{"username": "someuser", "key": "0123456789abcdef0123456789abcdef"}', "someuser"),
    ('{"username": "someuser", "key": "0123456789abcdef0123456789abcdef"}', "0123456789abcdef0123456789abcdef"),
    ("Authorization: Bearer eyJhbGciOi.abc.def", "eyJhbGciOi.abc.def"),
    ("api_key=abcd1234 password: hunter2", "hunter2"),
], ids=["legacy-hex-key", "kgat-token", "kgat-in-url", "json-username", "json-key", "bearer", "password"])
def test_credential_shapes_are_redacted(text, secret):
    assert secret not in bd.sanitize_error_text(text)


def test_study_uids_and_paths_stay_readable_after_sanitizing():
    text = "not found: train_series/1.2.826.0.1.3680043.8.498.10045003229099053869093324292195817260/1.2.3/7.dcm"
    assert bd.sanitize_error_text(text) == text
    assert bd.sanitize_error_text("Key error while parsing KeyError: 'x'").startswith("Key error while parsing KeyError")


# ------------------------------------------------------------------------------ helpers for study-level tests

def make_study(tmp_path, count, study=STUDY, series=SERIES):
    files = [f"train_series/{study}/{series}/{i}.dcm" for i in range(1, count + 1)]
    return {study: {series: files}}, tmp_path / "raw"


def failing(category_message, returncode=1):
    def fake(file_path, destination):
        message = bd.sanitize_error_text(category_message)
        return False, bd.DownloadFailure(file_path, message, bd.classify_error(message), 1, returncode)
    return fake


def succeeding(file_path, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(VALID_DICOM)
    return True, None


# ------------------------------------------------------------------------------ 7-8: aggregation and examples

def test_failure_categories_are_aggregated(monkeypatch, tmp_path):
    labeled, root = make_study(tmp_path, 12)
    messages = ["429 Too Many Requests"] * 6 + ["401 Unauthorized"] * 3 + ["Read timed out"] * 2 + ["weird"]

    def fake(file_path, destination):
        return failing(messages[int(destination.stem) - 1])(file_path, destination)

    monkeypatch.setattr(bd, "download_file_with_retry", fake)
    result = bd.download_studies_concurrent([STUDY], labeled, root, max_workers=2, breaker=bd.CircuitBreaker(10_000))
    assert result["total_failed"] == 12
    assert dict(result["failure_categories"]) == {
        bd.CATEGORY_RATE_LIMIT: 6, bd.CATEGORY_AUTH: 3, bd.CATEGORY_TIMEOUT: 2, bd.CATEGORY_UNKNOWN: 1,
    }


def test_representative_examples_are_limited_to_five(monkeypatch, tmp_path, capsys):
    labeled, root = make_study(tmp_path, 40)
    monkeypatch.setattr(bd, "download_file_with_retry", failing("429 Too Many Requests"))
    result = bd.download_studies_concurrent([STUDY], labeled, root, max_workers=2, breaker=bd.CircuitBreaker(10_000))
    assert result["total_failed"] == 40
    assert len(result["failure_examples"]) == bd.MAX_FAILURE_EXAMPLES == 5
    capsys.readouterr()
    bd.print_failure_report(result)
    out = capsys.readouterr().out
    assert "Failure categories:" in out
    assert f"{bd.CATEGORY_RATE_LIMIT}: 40" in out
    assert out.count("returncode=1") == 5
    assert out.count("attempts=1") == 5


def test_failure_report_is_silent_without_failures(capsys):
    bd.print_failure_report({"failure_categories": {}, "failure_examples": []})
    assert capsys.readouterr().out == ""


# ------------------------------------------------------------------------------ 9-10: circuit breaker

def test_counter_resets_on_success_unit():
    breaker = bd.CircuitBreaker(20)
    for _ in range(19):
        assert breaker.record(False) is False
    assert breaker.record(True) is False and breaker.consecutive_failures == 0
    for _ in range(19):
        assert breaker.record(False) is False
    assert breaker.record(False) is True and breaker.tripped


def test_default_threshold_is_twenty():
    assert bd.CIRCUIT_BREAKER_THRESHOLD == 20 and bd.CircuitBreaker().threshold == 20


def test_success_in_the_middle_prevents_tripping_during_a_study(monkeypatch, tmp_path):
    """19 failures, one success, 19 failures: never 20 in a row, so every file is attempted."""
    labeled, root = make_study(tmp_path, 39)
    pattern = [False] * 19 + [True] + [False] * 19
    lock, state = threading.Lock(), {"n": 0}

    def fake(file_path, destination):
        with lock:
            index = state["n"]
            state["n"] += 1
        REAL_SLEEP(0.01)  # keeps completion order == call order for a single worker
        if pattern[index]:
            return succeeding(file_path, destination)
        return failing("429 Too Many Requests")(file_path, destination)

    monkeypatch.setattr(bd, "download_file_with_retry", fake)
    result = bd.download_studies_concurrent([STUDY], labeled, root, max_workers=1)
    assert result["circuit_breaker_tripped"] is False
    assert result["total_failed"] == 38 and result["total_downloaded"] == 1 and result["total_not_attempted"] == 0


def test_circuit_breaker_activates_and_stops_new_downloads(monkeypatch, tmp_path):
    labeled, root = make_study(tmp_path, 60)
    calls = {"n": 0}
    lock = threading.Lock()

    def fake(file_path, destination):
        with lock:
            calls["n"] += 1
        REAL_SLEEP(0.01)
        return failing("429 Too Many Requests")(file_path, destination)

    monkeypatch.setattr(bd, "download_file_with_retry", fake)
    result = bd.download_studies_concurrent([STUDY], labeled, root, max_workers=2)
    study = result["studies"][STUDY]
    assert result["circuit_breaker_tripped"] is True
    assert study["failed"] >= bd.CIRCUIT_BREAKER_THRESHOLD
    assert study["not_attempted"] > 0
    assert study["failed"] + study["not_attempted"] == 60  # every queued file is accounted for
    assert calls["n"] == study["failed"] < 60  # cancelled files were never handed to Kaggle


def test_tripped_breaker_leaves_later_studies_unstarted(monkeypatch, tmp_path):
    labeled = {}
    for n in (1, 2, 3):
        labeled.update(make_study(tmp_path, 30, study=f"{STUDY}.{n}", series=f"{SERIES}.{n}")[0])
    root = tmp_path / "raw"

    def fake(file_path, destination):
        REAL_SLEEP(0.005)
        return failing("503 Service Unavailable")(file_path, destination)

    monkeypatch.setattr(bd, "download_file_with_retry", fake)
    studies = sorted(labeled)
    result = bd.download_studies_concurrent(studies, labeled, root, max_workers=2)
    assert result["circuit_breaker_tripped"] is True
    assert list(result["studies"]) == studies[:1]
    assert result["studies_not_started"] == studies[1:]
    assert not (root / studies[1]).exists() and not (root / studies[2]).exists()


def test_main_reports_breaker_and_exits_nonzero(monkeypatch, tmp_path, capsys):
    labeled, _ = make_study(tmp_path, 40)
    cache = tmp_path / "cache.json"
    cache.write_text(__import__("json").dumps(labeled))
    monkeypatch.setattr(bd, "CACHE_FILE", cache)
    monkeypatch.setattr(bd, "load_labeled_studies", lambda data_dir: {STUDY})
    monkeypatch.setattr(bd, "build_manifest", lambda labeled_files, data_dir: {})

    def fake(file_path, destination):
        REAL_SLEEP(0.005)
        return failing("429 Too Many Requests")(file_path, destination)

    monkeypatch.setattr(bd, "download_file_with_retry", fake)
    code = bd.main(["--data-dir", str(tmp_path), "--max-workers", "2"])
    out = capsys.readouterr().out
    assert code == 1
    assert "Circuit breaker triggered after 20 consecutive permanent download failures." in out
    assert "Failure categories:" in out and bd.CATEGORY_RATE_LIMIT in out
    assert out.index("BULK DOWNLOAD SUMMARY") < out.index("Circuit breaker triggered")


def test_main_exit_code_is_zero_when_nothing_fails(monkeypatch, tmp_path, capsys):
    labeled, _ = make_study(tmp_path, 5)
    cache = tmp_path / "cache.json"
    cache.write_text(__import__("json").dumps(labeled))
    monkeypatch.setattr(bd, "CACHE_FILE", cache)
    monkeypatch.setattr(bd, "load_labeled_studies", lambda data_dir: {STUDY})
    monkeypatch.setattr(bd, "build_manifest", lambda labeled_files, data_dir: {})
    monkeypatch.setattr(bd, "download_file_with_retry", succeeding)
    assert bd.main(["--data-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "Failure categories:" not in out and "Circuit breaker" not in out


# ------------------------------------------------------------------------------ 11-12: unchanged behavior

def test_skip_existing_behavior_is_unchanged(kaggle, tmp_path):
    labeled, root = make_study(tmp_path, 2)
    existing = root / STUDY / SERIES / "1.dcm"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(VALID_DICOM + b"marker")
    before = existing.read_bytes()
    fake = kaggle("ok")
    stats = bd.download_single_study(STUDY, labeled[STUDY], root, max_workers=1)
    assert stats["skipped"] == 1 and stats["downloaded"] == 1 and stats["failed"] == 0
    assert fake.calls == 1  # only the missing file was requested
    assert existing.read_bytes() == before  # the valid existing file was not touched
    assert (root / STUDY / SERIES / "2.dcm").read_bytes() == VALID_DICOM
    assert stats["series"][SERIES]["final_valid"] == 2


def test_all_existing_valid_files_means_no_kaggle_calls(kaggle, tmp_path):
    labeled, root = make_study(tmp_path, 3)
    for name in ("1.dcm", "2.dcm", "3.dcm"):
        (root / STUDY / SERIES).mkdir(parents=True, exist_ok=True)
        (root / STUDY / SERIES / name).write_bytes(VALID_DICOM)
    fake = kaggle("ok")
    stats = bd.download_single_study(STUDY, labeled[STUDY], root, max_workers=2)
    assert stats["skipped"] == 3 and stats["downloaded"] == 0 and fake.calls == 0


def test_valid_dicom_is_staged_then_moved_into_place(kaggle, tmp_path):
    fake = kaggle("ok")
    success, failure = download(tmp_path, "9.dcm")
    destination = tmp_path / STUDY / SERIES / "9.dcm"
    assert success is True and failure is None and fake.calls == 1
    assert destination.read_bytes() == VALID_DICOM and bd.is_valid_dicom(destination)
    assert [p.name for p in destination.parent.iterdir()] == ["9.dcm"]  # nothing else left behind


@pytest.mark.parametrize("outcome,message", [
    ("not-dicom", "Downloaded file is not valid DICOM"),
    ("empty", "No .dcm file in download"),
])
def test_invalid_download_never_reaches_destination(kaggle, tmp_path, outcome, message):
    kaggle(outcome)
    success, failure = download(tmp_path, "9.dcm")
    assert success is False
    assert failure.message == message and failure.category == bd.CATEGORY_DICOM
    assert failure.returncode == 0 and failure.attempts == 1
    assert not (tmp_path / STUDY / SERIES / "9.dcm").exists()

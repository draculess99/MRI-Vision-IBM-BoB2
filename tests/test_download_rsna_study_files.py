import importlib.util
import io
import subprocess
from pathlib import Path

import pytest

pydicom = pytest.importorskip("pydicom")
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_rsna_study_files.py"
spec = importlib.util.spec_from_file_location("download_rsna_study_files", SCRIPT)
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)

STUDY, SERIES = "1.2.3", "1.2.4"


def _dicom_bytes():
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset("mem.dcm", {}, file_meta=meta, preamble=b"\0" * 128)
    ds.Rows = ds.Columns = 4
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = bytes(32)
    buffer = io.BytesIO()
    ds.save_as(buffer)
    return buffer.getvalue()


DICOM_BYTES = _dicom_bytes()
OK = ("payload", DICOM_BYTES)
RATE_LIMITED = ("fail", "HTTP 429 Too Many Requests")
CONNECTION_DROPPED = ("fail", "Remote end closed connection without response")


class FakeKaggle:
    """Stands in for subprocess.run; each call consumes one scripted action, then defaults to OK."""

    def __init__(self, actions=()):
        self.actions = list(actions)
        self.commands = []

    @property
    def requested(self):
        return [command[command.index("-f") + 1] for command in self.commands]

    def __call__(self, command, **kwargs):
        self.commands.append(command)
        staging = Path(command[command.index("-p") + 1])
        kind, value = self.actions.pop(0) if self.actions else OK
        if kind == "fail":
            return subprocess.CompletedProcess(command, 1, "", value)
        (staging / Path(self.requested[-1]).name).write_bytes(value)
        return subprocess.CompletedProcess(command, 0, "", "")


@pytest.fixture(autouse=True)
def forbid_real_kaggle(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("test attempted a real Kaggle/subprocess call")
    monkeypatch.setattr(downloader.subprocess, "run", refuse)


@pytest.fixture
def sleeps(monkeypatch):
    recorded = []
    monkeypatch.setattr(downloader.time, "sleep", recorded.append)
    return recorded


@pytest.fixture
def install_kaggle(monkeypatch):
    def install(*actions):
        fake = FakeKaggle(actions)
        monkeypatch.setattr(downloader.subprocess, "run", fake)
        return fake
    return install


def make_names(count):
    return [f"train_series/{STUDY}/{SERIES}/1.2.5.{i}.dcm" for i in range(count)]


def setup_run(tmp_path, count=1):
    names = make_names(count)
    listing = tmp_path / "files.txt"
    listing.write_text("\n".join(names) + "\n", encoding="utf-8")
    root = tmp_path / "raw"
    return names, listing, root


def run(listing, root, *extra):
    return downloader.main(["--input", str(listing), "--output-root", str(root), *extra])


def destination(root, name):
    return downloader.destination_for(name, root)


def snapshot(root):
    return sorted((str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns)
                  for p in root.rglob("*"))


def test_downloads_single_file_via_kaggle_cli_and_validates_result(tmp_path, install_kaggle, sleeps):
    names, listing, root = setup_run(tmp_path)
    kaggle = install_kaggle()
    assert run(listing, root) == 0
    assert kaggle.commands[0][:5] == ["kaggle", "competitions", "download", downloader.COMPETITION, "-f"]
    assert kaggle.requested == names
    assert destination(root, names[0]).read_bytes() == DICOM_BYTES
    assert not (root / ".partial").exists()
    assert sleeps == []


def test_skips_existing_non_empty_file(tmp_path, install_kaggle, sleeps, capsys):
    names, listing, root = setup_run(tmp_path, count=2)
    existing = destination(root, names[0])
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"already here")
    kaggle = install_kaggle()
    assert run(listing, root) == 0
    assert kaggle.requested == [names[1]]
    assert existing.read_bytes() == b"already here"
    assert destination(root, names[1]).read_bytes() == DICOM_BYTES
    assert "skipped (already present): 1" in capsys.readouterr().out


def test_zero_byte_file_is_downloaded_again(tmp_path, install_kaggle, sleeps):
    names, listing, root = setup_run(tmp_path)
    target = destination(root, names[0])
    target.parent.mkdir(parents=True)
    target.write_bytes(b"")
    kaggle = install_kaggle()
    assert run(listing, root) == 0
    assert kaggle.requested == names
    assert target.read_bytes() == DICOM_BYTES


def test_transient_failure_is_retried_then_succeeds(tmp_path, install_kaggle, sleeps):
    names, listing, root = setup_run(tmp_path)
    kaggle = install_kaggle(CONNECTION_DROPPED, OK)
    assert run(listing, root, "--retries", "3", "--backoff", "2") == 0
    assert kaggle.requested == names * 2
    assert sleeps == [2.0]
    assert destination(root, names[0]).read_bytes() == DICOM_BYTES
    assert not (root / ".partial").exists()


def test_http_429_backs_off_exponentially(tmp_path, install_kaggle, sleeps):
    names, listing, root = setup_run(tmp_path)
    kaggle = install_kaggle(RATE_LIMITED, RATE_LIMITED, OK)
    assert run(listing, root, "--retries", "4", "--backoff", "1.5") == 0
    assert len(kaggle.commands) == 3
    assert sleeps == [1.5, 3.0]
    assert destination(root, names[0]).is_file()


def test_http_429_exhausting_retries_reports_failure_without_final_sleep(tmp_path, install_kaggle, sleeps, capsys):
    names, listing, root = setup_run(tmp_path)
    kaggle = install_kaggle(RATE_LIMITED, RATE_LIMITED)
    assert run(listing, root, "--retries", "2", "--backoff", "1") == 1
    assert len(kaggle.commands) == 2
    assert sleeps == [1.0]
    assert not destination(root, names[0]).exists()
    assert "failed: 1" in capsys.readouterr().out


def test_non_transient_error_stops_without_retrying(tmp_path, install_kaggle, sleeps):
    names, listing, root = setup_run(tmp_path, count=3)
    kaggle = install_kaggle(("fail", "403 Forbidden: you must accept the competition rules"))
    assert run(listing, root) == 2
    assert len(kaggle.commands) == 1
    assert sleeps == []


@pytest.mark.parametrize("payload", [b"this is not a DICOM file", b""], ids=["non-dicom", "empty"])
def test_invalid_download_is_rejected_and_never_reaches_destination(tmp_path, install_kaggle, sleeps, payload):
    names, listing, root = setup_run(tmp_path)
    kaggle = install_kaggle(("payload", payload), ("payload", payload))
    assert run(listing, root, "--retries", "2", "--backoff", "1") == 1
    assert len(kaggle.commands) == 2
    assert not destination(root, names[0]).exists()
    assert not (root / ".partial").exists()


def test_invalid_download_is_replaced_by_valid_retry(tmp_path, install_kaggle, sleeps):
    names, listing, root = setup_run(tmp_path)
    install_kaggle(("payload", b"truncated garbage"), OK)
    assert run(listing, root, "--retries", "2", "--backoff", "1") == 0
    assert destination(root, names[0]).read_bytes() == DICOM_BYTES


@pytest.mark.parametrize("root_exists", [False, True], ids=["fresh-root", "populated-root"])
def test_dry_run_makes_no_network_calls_or_filesystem_changes(tmp_path, sleeps, capsys, root_exists):
    names, listing, root = setup_run(tmp_path, count=3)
    if root_exists:
        kept = destination(root, names[0])
        kept.parent.mkdir(parents=True)
        kept.write_bytes(b"already here")
    before = snapshot(root) if root_exists else None
    assert run(listing, root, "--dry-run") == 0  # forbid_real_kaggle fails the test on any subprocess call
    assert sleeps == []
    if root_exists:
        assert snapshot(root) == before
    else:
        assert not root.exists()
    out = capsys.readouterr().out
    assert "[dry run]" in out
    assert out.count("would download") == (2 if root_exists else 3)


def test_limit_caps_downloads_and_later_runs_continue_with_next_files(tmp_path, install_kaggle, sleeps, capsys):
    names, listing, root = setup_run(tmp_path, count=5)
    kaggle = install_kaggle()
    assert run(listing, root, "--limit", "2") == 0
    assert kaggle.requested == names[:2]
    assert [destination(root, n).exists() for n in names] == [True, True, False, False, False]
    assert "not attempted (pending beyond --limit): 3" in capsys.readouterr().out

    assert run(listing, root, "--limit", "2") == 0
    assert kaggle.requested == names[:4]
    assert [destination(root, n).exists() for n in names] == [True, True, True, True, False]


def test_limit_applies_to_dry_run_and_must_be_positive(tmp_path, capsys):
    names, listing, root = setup_run(tmp_path, count=5)
    assert run(listing, root, "--dry-run", "--limit", "2") == 0
    out = capsys.readouterr().out
    assert "selected this run: 2" in out
    assert out.count("would download") == 2
    with pytest.raises(SystemExit) as excinfo:
        run(listing, root, "--limit", "0")
    assert excinfo.value.code == 2
    assert not root.exists()


def test_rejects_unexpected_names_in_input_list(tmp_path, capsys):
    listing = tmp_path / "files.txt"
    listing.write_text("train_series/../../escape.dcm\n", encoding="utf-8")
    assert run(listing, tmp_path / "raw") == 2
    assert "unexpected file name" in capsys.readouterr().err

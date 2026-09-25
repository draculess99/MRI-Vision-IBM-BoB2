"""Real dataset access is opt-in; the default suite is entirely synthetic."""

import pytest

OPT_IN = {
    "mrnet": ("--run-mrnet", "Local MRNet check requires --run-mrnet"),
    "rsna_real": ("--run-rsna", "Downloaded RSNA study check requires --run-rsna"),
}


def pytest_addoption(parser):
    parser.addoption("--run-mrnet", action="store_true", default=False,
                     help="Run optional read-only checks against the local MRNet dataset")
    parser.addoption("--run-rsna", action="store_true", default=False,
                     help="Run optional read-only checks against the locally downloaded RSNA study")


def pytest_configure(config):
    config.addinivalue_line("markers", "mrnet: optional read-only local MRNet integration check")
    config.addinivalue_line("markers", "rsna_real: optional read-only check against the downloaded RSNA study")
    config.addinivalue_line("markers", "rsna: read-only RSNA integration check; skips cleanly if local data is absent")


def pytest_collection_modifyitems(config, items):
    for marker, (option, reason) in OPT_IN.items():
        if not config.getoption(option):
            skip = pytest.mark.skip(reason=reason)
            for item in items:
                if marker in item.keywords:
                    item.add_marker(skip)

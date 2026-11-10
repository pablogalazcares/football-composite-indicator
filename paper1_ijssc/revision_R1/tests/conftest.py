from __future__ import annotations

import sys
from pathlib import Path

import pytest

PIPE = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPE) not in sys.path:
    sys.path.insert(0, str(PIPE))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: usa datos reales o artefactos del pipeline (lento)")


@pytest.fixture(scope="session")
def cfg():
    import common
    return common.load_config()

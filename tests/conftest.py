"""Pytest configuration and shared fixtures for ai-inference-bench."""

import shutil
import tempfile
from pathlib import Path
import pytest


@pytest.fixture
def temp_results_dir():
    """Provides an isolated temporary directory for benchmark output artifacts."""
    temp_dir = tempfile.mkdtemp(prefix="bench_test_results_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)

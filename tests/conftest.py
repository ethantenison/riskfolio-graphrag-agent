"""Shared pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture()
def tmp_source_dir(tmp_path):
    """Create a temporary directory with sample source files for ingestion tests."""
    py_file = tmp_path / "sample.py"
    py_file.write_text("def hello():\n    return 'world'\n")
    md_file = tmp_path / "README.md"
    md_file.write_text("# Sample\n\nThis is a test.\n")
    return tmp_path

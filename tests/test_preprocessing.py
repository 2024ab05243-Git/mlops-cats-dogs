import pytest
from pathlib import Path
from src.data import preprocess


def test_create_dirs(tmp_path, monkeypatch):
    # Redirect PROCESSED_DIR to temporary path
    monkeypatch.setattr(preprocess, "PROCESSED_DIR", tmp_path)

    preprocess.create_dirs()

    # Expected structure
    expected_paths = [
        tmp_path / "train" / "cats",
        tmp_path / "train" / "dogs",
        tmp_path / "val" / "cats",
        tmp_path / "val" / "dogs",
        tmp_path / "test" / "cats",
        tmp_path / "test" / "dogs",
    ]

    for path in expected_paths:
        assert path.exists()
        assert path.is_dir()
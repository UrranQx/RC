from __future__ import annotations

import importlib
from pathlib import Path

import conn2res.connectivity as connectivity


def reload_connectivity(monkeypatch, *, root: Path, data_dir: Path | None = None):
    monkeypatch.setenv("CONN2RES_ROOT", str(root))
    if data_dir is None:
        monkeypatch.delenv("CONN2RES_DATA_DIR", raising=False)
    else:
        monkeypatch.setenv("CONN2RES_DATA_DIR", str(data_dir))
    return importlib.reload(connectivity)


def test_default_data_dir_uses_repository_data_directory(monkeypatch, tmp_path):
    module = reload_connectivity(monkeypatch, root=tmp_path)

    assert Path(module.DATA_DIR) == tmp_path / "data" / "human"


def test_data_dir_environment_variable_overrides_repository_root(monkeypatch, tmp_path):
    custom_data_dir = tmp_path / "custom-connectomes"

    module = reload_connectivity(
        monkeypatch,
        root=tmp_path / "repository",
        data_dir=custom_data_dir,
    )

    assert Path(module.DATA_DIR) == custom_data_dir

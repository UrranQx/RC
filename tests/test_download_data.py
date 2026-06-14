from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "download_data.py"
SPEC = importlib.util.spec_from_file_location("download_data", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
download_data = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = download_data
SPEC.loader.exec_module(download_data)

TEST_SHAPES = {
    "connectivity.npy": (3, 3, 2),
    "coords.npy": (3, 3),
}


def write_dataset(directory: Path, shapes=TEST_SHAPES) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for filename, shape in shapes.items():
        np.save(directory / filename, np.zeros(shape, dtype=np.float32))


def write_archive(path: Path, source: Path, *, extra_member: str | None = None) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for filename in TEST_SHAPES:
            archive.write(source / filename, f"data/human/{filename}")
        if extra_member is not None:
            archive.writestr(extra_member, b"unexpected")


def test_verify_checksum_accepts_matching_file_and_rejects_mismatch(tmp_path):
    archive = tmp_path / "data.zip"
    archive.write_bytes(b"connectome data")
    checksum = f"md5:{hashlib.md5(archive.read_bytes()).hexdigest()}"

    download_data.verify_checksum(archive, checksum)

    with pytest.raises(download_data.DownloadError, match="checksum"):
        download_data.verify_checksum(archive, "md5:00000000000000000000000000000000")


def test_safe_extract_extracts_only_expected_human_data(tmp_path):
    source = tmp_path / "source"
    write_dataset(source)
    archive = tmp_path / "data.zip"
    write_archive(archive, source)
    output_dir = tmp_path / "output"

    download_data.safe_extract(archive, output_dir, expected_shapes=TEST_SHAPES)

    assert sorted(path.name for path in output_dir.glob("*.npy")) == sorted(TEST_SHAPES)


def test_safe_extract_ignores_unexpected_safe_members(tmp_path):
    source = tmp_path / "source"
    write_dataset(source)
    archive = tmp_path / "data.zip"
    write_archive(archive, source, extra_member="data/drosophila/connectivity.npy")
    output_dir = tmp_path / "output"

    download_data.safe_extract(archive, output_dir, expected_shapes=TEST_SHAPES)

    assert not (tmp_path / "drosophila").exists()
    assert sorted(path.name for path in output_dir.glob("*.npy")) == sorted(TEST_SHAPES)


def test_safe_extract_rejects_traversal_members(tmp_path):
    source = tmp_path / "source"
    write_dataset(source)
    archive = tmp_path / "traversal.zip"
    write_archive(archive, source, extra_member="../../outside.txt")

    with pytest.raises(download_data.DownloadError, match="unexpected"):
        download_data.safe_extract(
            archive,
            tmp_path / "output",
            expected_shapes=TEST_SHAPES,
        )


def test_validate_dataset_checks_names_and_shapes(tmp_path):
    write_dataset(tmp_path)
    download_data.validate_dataset(tmp_path, expected_shapes=TEST_SHAPES)

    np.save(tmp_path / "coords.npy", np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(download_data.DownloadError, match="coords.npy"):
        download_data.validate_dataset(tmp_path, expected_shapes=TEST_SHAPES)

    np.save(tmp_path / "coords.npy", np.zeros(TEST_SHAPES["coords.npy"]))
    np.save(tmp_path / "unexpected.npy", np.zeros((1,)))
    with pytest.raises(download_data.DownloadError, match="unexpected"):
        download_data.validate_dataset(tmp_path, expected_shapes=TEST_SHAPES)


def test_fetch_download_spec_uses_zenodo_metadata_without_real_network():
    payload = {
        "files": [
            {
                "key": "data.zip",
                "checksum": "md5:abc",
                "links": {"self": "https://example.invalid/data.zip"},
            }
        ]
    }

    def fake_opener(_url):
        return io.BytesIO(json.dumps(payload).encode())

    spec = download_data.fetch_download_spec(opener=fake_opener)

    assert spec.url == "https://example.invalid/data.zip"
    assert spec.checksum == "md5:abc"


def test_check_mode_validates_local_data_without_network(tmp_path):
    write_dataset(tmp_path)

    def fail_if_called(_url):
        raise AssertionError("network request was not expected")

    download_data.run(
        output_dir=tmp_path,
        check=True,
        opener=fail_if_called,
        expected_shapes=TEST_SHAPES,
    )


def test_valid_dataset_is_not_overwritten_without_force(tmp_path):
    write_dataset(tmp_path)
    original = (tmp_path / "coords.npy").read_bytes()

    def fail_if_called(_url):
        raise AssertionError("network request was not expected")

    download_data.run(
        output_dir=tmp_path,
        opener=fail_if_called,
        expected_shapes=TEST_SHAPES,
    )

    assert (tmp_path / "coords.npy").read_bytes() == original

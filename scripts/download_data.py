#!/usr/bin/env python3
"""Download and validate the connectome data used by the experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Callable
from urllib.request import urlopen

RECORD_ID = "10205004"
RECORD_API_URL = f"https://zenodo.org/api/records/{RECORD_ID}"
ARCHIVE_NAME = "data.zip"
ARCHIVE_PREFIX = PurePosixPath("data/human")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "human"

EXPECTED_SHAPES = {
    "connectivity.npy": (1015, 1015, 70),
    "consensus_0.npy": (1015, 1015),
    "consensus_1.npy": (1015, 1015),
    "consensus_2.npy": (1015, 1015),
    "consensus_3.npy": (1015, 1015),
    "consensus_4.npy": (1015, 1015),
    "consensus_5.npy": (1015, 1015),
    "coords.npy": (1015, 3),
    "cortical.npy": (1015,),
    "hemiid.npy": (1015,),
    "rsn_mapping.npy": (1015,),
}

Opener = Callable[[str], BinaryIO]


class DownloadError(RuntimeError):
    """Raised when downloaded or local data fail validation."""


@dataclass(frozen=True)
class DownloadSpec:
    """Metadata required to download and verify the Zenodo archive."""

    url: str
    checksum: str


def fetch_download_spec(opener: Opener = urlopen) -> DownloadSpec:
    """Return the URL and checksum for data.zip from the Zenodo record."""
    with opener(RECORD_API_URL) as response:
        metadata = json.load(response)

    for file_metadata in metadata.get("files", []):
        if file_metadata.get("key") == ARCHIVE_NAME:
            url = file_metadata.get("links", {}).get("self")
            checksum = file_metadata.get("checksum")
            if url and checksum:
                return DownloadSpec(url=url, checksum=checksum)

    raise DownloadError(f"{ARCHIVE_NAME} is missing from Zenodo record {RECORD_ID}")


def verify_checksum(path: Path, expected_checksum: str) -> None:
    """Verify a file checksum in the ``algorithm:hex-digest`` format."""
    try:
        algorithm, expected_digest = expected_checksum.split(":", maxsplit=1)
        digest = hashlib.new(algorithm)
    except (ValueError, TypeError) as exc:
        raise DownloadError(f"Unsupported checksum: {expected_checksum}") from exc

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    if digest.hexdigest().lower() != expected_digest.lower():
        raise DownloadError(f"Archive checksum mismatch for {path.name}")


def download_archive(
    spec: DownloadSpec, destination: Path, opener: Opener = urlopen
) -> None:
    """Download an archive and verify it before returning."""
    with opener(spec.url) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)
    verify_checksum(destination, spec.checksum)


def safe_extract(
    archive_path: Path,
    output_dir: Path,
    *,
    expected_shapes: Mapping[str, tuple[int, ...]] = EXPECTED_SHAPES,
) -> None:
    """Extract exactly the expected ``data/human`` NumPy files."""
    expected_members = {
        str(ARCHIVE_PREFIX / filename): filename for filename in expected_shapes
    }
    found_members: dict[str, zipfile.ZipInfo] = {}

    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            member_path = PurePosixPath(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise DownloadError(
                    f"Archive contains unexpected member: {member.filename}"
                )
            if member.is_dir():
                continue
            filename = expected_members.get(str(member_path))
            if filename is None:
                continue
            if filename in found_members:
                raise DownloadError(
                    f"Archive contains unexpected member: {member.filename}"
                )
            found_members[filename] = member

        missing = sorted(set(expected_shapes) - set(found_members))
        if missing:
            raise DownloadError(
                f"Archive is missing expected files: {', '.join(missing)}"
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        for filename, member in found_members.items():
            with (
                archive.open(member) as source,
                (output_dir / filename).open("wb") as target,
            ):
                shutil.copyfileobj(source, target)


def validate_dataset(
    data_dir: Path,
    *,
    expected_shapes: Mapping[str, tuple[int, ...]] = EXPECTED_SHAPES,
) -> None:
    """Validate expected filenames and array shapes without loading array data."""
    import numpy as np

    data_dir = Path(data_dir)
    present_npy = {path.name for path in data_dir.glob("*.npy")}
    unexpected = sorted(present_npy - set(expected_shapes))
    missing = sorted(set(expected_shapes) - present_npy)
    if unexpected:
        raise DownloadError(
            f"Data directory contains unexpected files: {', '.join(unexpected)}"
        )
    if missing:
        raise DownloadError(
            f"Data directory is missing expected files: {', '.join(missing)}"
        )

    for filename, expected_shape in expected_shapes.items():
        path = data_dir / filename
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except Exception as exc:
            raise DownloadError(f"Cannot read {filename}: {exc}") from exc
        if array.shape != expected_shape:
            raise DownloadError(
                f"{filename} has shape {array.shape}, expected {expected_shape}"
            )


def run(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    check: bool = False,
    force: bool = False,
    opener: Opener = urlopen,
    expected_shapes: Mapping[str, tuple[int, ...]] = EXPECTED_SHAPES,
) -> None:
    """Check local data or download a validated copy from Zenodo."""
    output_dir = Path(output_dir).resolve()

    if check:
        validate_dataset(output_dir, expected_shapes=expected_shapes)
        print(f"Data check passed: {output_dir}")
        return

    try:
        validate_dataset(output_dir, expected_shapes=expected_shapes)
    except DownloadError:
        pass
    else:
        if not force:
            print(f"Valid data already exist: {output_dir}")
            return

    existing_npy = list(output_dir.glob("*.npy"))
    if existing_npy and not force:
        raise DownloadError(
            f"{output_dir} contains an incomplete or invalid dataset; use --force"
        )

    spec = fetch_download_spec(opener=opener)
    with tempfile.TemporaryDirectory(prefix="conn2res-data-") as temp_directory:
        temp_dir = Path(temp_directory)
        archive_path = temp_dir / ARCHIVE_NAME
        staging_dir = temp_dir / "human"
        download_archive(spec, archive_path, opener=opener)
        safe_extract(archive_path, staging_dir, expected_shapes=expected_shapes)
        validate_dataset(staging_dir, expected_shapes=expected_shapes)

        output_dir.mkdir(parents=True, exist_ok=True)
        if force:
            for path in output_dir.glob("*.npy"):
                path.unlink()
        for filename in expected_shapes:
            shutil.copy2(staging_dir / filename, output_dir / filename)

    validate_dataset(output_dir, expected_shapes=expected_shapes)
    print(f"Data downloaded and validated: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and validate connectome data from Zenodo."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate local data without using the network",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing or incomplete dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="target directory for the data/human NumPy files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        run(output_dir=args.output_dir, check=args.check, force=args.force)
    except DownloadError as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()

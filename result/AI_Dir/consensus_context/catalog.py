"""Discovery and integrity metadata for consensus source snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
from pathlib import Path
import re


_KINDS = ("CQBtw_Q_sector", "CQBtw_Q", "Q")
_DATE_SUFFIX = re.compile(r"_(?P<date>\d{4}-\d{2}-\d{2})\.csv$", re.IGNORECASE)


@dataclass(frozen=True)
class SnapshotFile:
    """A source CSV and its date and content digest."""

    kind: str
    as_of_date: date
    path: Path
    sha256: str


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of *path*, reading it incrementally."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _kind_from_name(stem: str) -> str:
    for kind in _KINDS:
        if stem == kind or stem.endswith(f"_{kind}") or f"_{kind}_" in stem:
            return kind
    raise ValueError(f"Unsupported consensus snapshot kind in filename: {stem!r}")


def discover_snapshots(data_root: Path) -> list[SnapshotFile]:
    """Discover dated consensus CSVs below *data_root*.

    Dates are taken only from the filename suffix. Every CSV must end in
    ``_YYYY-MM-DD.csv``; filesystem modification times are intentionally not
    consulted.
    """

    root = Path(data_root)
    snapshots: list[SnapshotFile] = []
    for path in sorted(root.rglob("*.csv")):
        match = _DATE_SUFFIX.search(path.name)
        if match is None:
            raise ValueError(f"CSV filename has no YYYY-MM-DD date: {path.name!r}")
        snapshots.append(
            SnapshotFile(
                kind=_kind_from_name(
                    path.stem[: -(len(match.group("date")) + 1)]
                ),
                as_of_date=date.fromisoformat(match.group("date")),
                path=path,
                sha256=sha256_file(path),
            )
        )
    return snapshots

"""Schema validation for company and sector consensus snapshots."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import pandas as pd

from .catalog import sha256_file


class SchemaError(ValueError):
    """Raised when a snapshot cannot provide unique canonical keys."""


@dataclass
class ValidationResult:
    """Rows accepted and rejected during schema validation, plus diagnostics."""

    valid_rows: pd.DataFrame
    rejected_rows: pd.DataFrame
    issues: list[dict]


_METADATA_FILENAMES = (
    "metadata_Q.txt",
    "metadata_CQBtw_Q.txt",
    "metadata_CQBtw_Q_sector.txt",
    "metadata_total_ts.txt",
)


def _present(values: pd.Series) -> pd.Series:
    """Return whether values are non-null and, for text, non-blank."""

    return values.notna() & values.astype("string").str.strip().ne("")


def _validate_snapshot(
    frame: pd.DataFrame,
    *,
    identifiers: tuple[str, ...],
    estimate: str,
) -> ValidationResult:
    """Reject unusable rows and ensure the surviving canonical keys are unique."""

    source = frame.copy()
    rejected = pd.Series(False, index=source.index, dtype=bool)
    issues: list[dict] = []

    for field in (*identifiers, estimate):
        if field not in source.columns:
            invalid = pd.Series(True, index=source.index, dtype=bool)
            reason = "missing required column"
        elif field == estimate:
            numeric = pd.to_numeric(source[field], errors="coerce")
            invalid = ~numeric.map(math.isfinite)
            reason = "unusable estimate"
        else:
            invalid = ~_present(source[field])
            reason = "missing required identifier"

        count = int((invalid & ~rejected).sum())
        if count:
            issues.append({"field": field, "row_count": count, "reason": reason})
        rejected |= invalid

    valid_rows = source.loc[~rejected].copy()
    rejected_rows = source.loc[rejected].copy()

    if not valid_rows.empty:
        duplicate = valid_rows.duplicated(list(identifiers), keep=False)
        if duplicate.any():
            keys = valid_rows.loc[duplicate, list(identifiers)].drop_duplicates()
            raise SchemaError(f"duplicate canonical keys: {keys.to_dict(orient='records')}")

    return ValidationResult(
        valid_rows=valid_rows,
        rejected_rows=rejected_rows,
        issues=issues,
    )


def validate_company_snapshot(frame: pd.DataFrame) -> ValidationResult:
    """Validate company rows keyed by ``Code × FY × CQBtw``."""

    return _validate_snapshot(
        frame,
        identifiers=("Code", "FY", "CQBtw"),
        estimate="EPS_Est",
    )


def validate_sector_snapshot(frame: pd.DataFrame) -> ValidationResult:
    """Validate sector rows keyed by ``Sector × FY × CQBtw``."""

    return _validate_snapshot(
        frame,
        identifiers=("Sector", "FY", "CQBtw"),
        estimate="earning_total",
    )


def metadata_provenance(metadata_root: Path) -> list[dict]:
    """Return required metadata files with their SHA-256 content digests."""

    root = Path(metadata_root)
    missing = [name for name in _METADATA_FILENAMES if not (root / name).is_file()]
    if missing:
        raise SchemaError(f"missing metadata files: {', '.join(missing)}")

    return [
        {"path": root / name, "sha256": sha256_file(root / name)}
        for name in _METADATA_FILENAMES
    ]

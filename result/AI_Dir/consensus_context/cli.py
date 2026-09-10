"""Command-line orchestration for consensus research-context artifacts."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import date, datetime, timezone
import hashlib
from pathlib import Path
import sys

import pandas as pd

from .alignment import align_company_snapshots, align_sector_snapshots
from .catalog import SnapshotFile, _kind_from_name, sha256_file
from .contracts import build_research_context, ensure_user_context
from .contribution import attribute_contributors
from .features import (
    MIN_FLAGGED_FORWARD_HORIZONS,
    MIN_MATCHED_COMPANIES,
    MIN_MATCHED_COVERAGE,
    build_sector_features,
)
from .serialization import write_pretty_json
from .validation import (
    ValidationResult,
    metadata_provenance,
    validate_company_snapshot,
    validate_sector_snapshot,
)


_COMPANY_KIND = "Q"
_SECTOR_KIND = "CQBtw_Q_sector"
_IDENTIFIER_DTYPES = {
    "Code": "string",
    "FY": "string",
    "Sector": "string",
    "LSector": "string",
}


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build-consensus-context",
        description="Build deterministic JSON facts from two consensus snapshots.",
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--metadata-root", required=True, type=Path)
    parser.add_argument("--old-date", required=True, type=date.fromisoformat)
    parser.add_argument("--new-date", required=True, type=date.fromisoformat)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top-contributors", type=_positive_integer, default=10)
    return parser


def _snapshot_for(data_root: Path, kind: str, as_of_date: date) -> SnapshotFile:
    date_text = as_of_date.isoformat()
    suffix = f"_{date_text}.csv"
    matches = sorted(
        path
        for path in Path(data_root).rglob(f"*{suffix}")
        if _kind_from_name(path.stem[: -(len(date_text) + 1)]) == kind
    )
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {kind} snapshot for {as_of_date.isoformat()}, "
            f"found {len(matches)}"
        )
    path = matches[0]
    return SnapshotFile(
        kind=kind,
        as_of_date=as_of_date,
        path=path,
        sha256=sha256_file(path),
    )


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(
            path, encoding="utf-8-sig", dtype=_IDENTIFIER_DTYPES
        )
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp949", dtype=_IDENTIFIER_DTYPES)


def _run_id(
    old_date: date,
    new_date: date,
    sources: Sequence[dict],
) -> str:
    digest = hashlib.sha256()
    for source in sorted(sources, key=lambda item: str(item["path"])):
        digest.update(str(source["path"]).encode("utf-8"))
        digest.update(str(source["sha256"]).encode("ascii"))
    return (
        f"consensus-{old_date.isoformat()}-{new_date.isoformat()}-"
        f"{digest.hexdigest()[:12]}"
    )


def _source_record(source: SnapshotFile) -> dict:
    return {
        "kind": source.kind,
        "as_of_date": source.as_of_date,
        "path": source.path,
        "sha256": source.sha256,
    }


def _validation_record(
    kind: str,
    as_of_date: date,
    result: ValidationResult,
) -> dict:
    return {
        "kind": kind,
        "as_of_date": as_of_date.isoformat(),
        "valid_rows": len(result.valid_rows),
        "rejected_rows": len(result.rejected_rows),
        "issues": result.issues,
    }


def _exclude_market_total(frame: pd.DataFrame) -> pd.DataFrame:
    """Exclude the ``00`` all-market aggregate from sector cross-sections."""

    sector_codes = frame["Sector"].astype("string").str.strip().str.zfill(2)
    return frame.loc[sector_codes.ne("00")].copy()


def _attach_sector_totals(
    company_panel: pd.DataFrame,
    sector_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Attach validated top-level aggregate totals for reconciliation."""

    total_columns = ["old_earning_total", "new_earning_total"]
    missing = [column for column in total_columns if column not in sector_panel]
    if missing:
        raise KeyError(f"missing aligned sector total columns: {', '.join(missing)}")

    top_level = sector_panel.loc[
        sector_panel["Sector"].astype("string").str.len().eq(2),
        ["Sector", "FY", "CQBtw", *total_columns],
    ].rename(columns={"Sector": "top_sector"})
    attributes = company_panel.attrs.copy()
    enriched = company_panel.merge(
        top_level,
        on=["top_sector", "FY", "CQBtw"],
        how="left",
        validate="many_to_one",
    )
    enriched.attrs = attributes
    return enriched


def build_context(
    data_root: Path,
    metadata_root: Path,
    old_date: date,
    new_date: date,
    output_dir: Path,
    top_n: int = 10,
) -> Path:
    """Build fact, editable-user, and manifest JSON files from two dates.

    Inputs are only opened for reading. An existing ``user_context.json`` is
    validated and retained byte-for-byte by :func:`ensure_user_context`.
    """

    if old_date >= new_date:
        raise ValueError("old_date must be earlier than new_date")
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n <= 0:
        raise ValueError("top_n must be a positive integer")

    selected = [
        _snapshot_for(Path(data_root), _COMPANY_KIND, old_date),
        _snapshot_for(Path(data_root), _COMPANY_KIND, new_date),
        _snapshot_for(Path(data_root), _SECTOR_KIND, old_date),
        _snapshot_for(Path(data_root), _SECTOR_KIND, new_date),
    ]
    company_old_source, company_new_source, sector_old_source, sector_new_source = (
        selected
    )

    company_old = validate_company_snapshot(_read_csv(company_old_source.path))
    company_new = validate_company_snapshot(_read_csv(company_new_source.path))
    sector_old = validate_sector_snapshot(_read_csv(sector_old_source.path))
    sector_new = validate_sector_snapshot(_read_csv(sector_new_source.path))

    company_panel = align_company_snapshots(
        company_old.valid_rows, company_new.valid_rows
    )
    sector_panel = align_sector_snapshots(
        _exclude_market_total(sector_old.valid_rows),
        _exclude_market_total(sector_new.valid_rows),
    )
    company_panel = _attach_sector_totals(company_panel, sector_panel)
    sectors = build_sector_features(company_panel, sector_panel)
    contributions = attribute_contributors(company_panel, top_n=top_n)
    for sector in sectors:
        sector["contributions"] = contributions.get(sector["sector"], {})

    provenance = [_source_record(source) for source in selected]
    provenance.extend(metadata_provenance(Path(metadata_root)))
    run_id = _run_id(old_date, new_date, provenance)
    validations = [
        _validation_record(_COMPANY_KIND, old_date, company_old),
        _validation_record(_COMPANY_KIND, new_date, company_new),
        _validation_record(_SECTOR_KIND, old_date, sector_old),
        _validation_record(_SECTOR_KIND, new_date, sector_new),
    ]
    research = build_research_context(
        run_id=run_id,
        generated_at=datetime.now(timezone.utc),
        old_date=old_date,
        new_date=new_date,
        provenance=provenance,
        methodology={
            "company_key": ["Code", "FY", "CQBtw"],
            "sector_key": ["Sector", "FY", "CQBtw"],
            "forward_horizons": [1, 2, 3],
            "forward_weights": {"1": 0.45, "2": 0.35, "3": 0.20},
            "breadth_tolerance": 0.001,
            "excluded_aggregate_sector_codes": ["00"],
            "contribution_formula": "(new EPS_Est - old EPS_Est) * old shares",
            "top_contributors_per_tail": top_n,
            "quality_gate_thresholds": {
                "minimum_matched_coverage": MIN_MATCHED_COVERAGE,
                "minimum_matched_companies": MIN_MATCHED_COMPANIES,
                "flagged_forward_horizons": MIN_FLAGGED_FORWARD_HORIZONS,
            },
        },
        sectors=sectors,
        validation_summary={
            "sources": validations,
            "valid_rows": sum(record["valid_rows"] for record in validations),
            "rejected_rows": sum(
                record["rejected_rows"] for record in validations
            ),
            "issues": [
                issue
                for record in validations
                for issue in record["issues"]
            ],
            "company_alignment": company_panel.attrs.get("coverage_totals", {}),
            "sector_alignment": sector_panel.attrs.get("coverage_totals", {}),
        },
    )

    destination = Path(output_dir)
    research_path = destination / "research_context.json"
    user_path = destination / "user_context.json"
    write_pretty_json(research_path, research)
    ensure_user_context(user_path, run_id)

    source_hashes = {
        Path(record["path"]).as_posix(): record["sha256"]
        for record in provenance
    }
    manifest = {
        "schema_version": "1.0",
        "run_id": run_id,
        "generated_at": research["generated_at"],
        "snapshot_dates": {
            "old": old_date.isoformat(),
            "new": new_date.isoformat(),
        },
        "research_context": research_path.as_posix(),
        "research_context_sha256": sha256_file(research_path),
        "user_context": user_path.as_posix(),
        "user_context_sha256": sha256_file(user_path),
        "source_sha256": source_hashes,
    }
    write_pretty_json(destination / "build_manifest.json", manifest)
    return research_path


def main(argv: Sequence[str] | None = None) -> int:
    """Run the builder and return a conventional process exit status."""

    parser = _argument_parser()
    try:
        arguments = parser.parse_args(argv)
        build_context(
            data_root=arguments.data_root,
            metadata_root=arguments.metadata_root,
            old_date=arguments.old_date,
            new_date=arguments.new_date,
            output_dir=arguments.output,
            top_n=arguments.top_contributors,
        )
    except SystemExit as exc:
        return int(exc.code)
    except (OSError, ValueError, KeyError) as exc:
        print(f"build-consensus-context: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from datetime import date

from consensus_context.catalog import discover_snapshots, sha256_file
from consensus_context.validation import (
    SchemaError,
    metadata_provenance,
    validate_company_snapshot,
    validate_sector_snapshot,
)
import pytest
import pandas as pd


def test_catalog_uses_filename_date_and_hash(tmp_path):
    source = tmp_path / "mixed_model_Q_2026-09-08.csv"
    source.write_text("Code,FY,CQBtw,EPS_Est\nA,2026Q4,1,2.0\n", encoding="utf-8")
    items = discover_snapshots(tmp_path)
    assert items[0].as_of_date == date(2026, 9, 8)
    assert items[0].sha256 == sha256_file(source)


def test_catalog_recognizes_explicit_kinds(tmp_path):
    for kind in ("Q", "CQBtw_Q", "CQBtw_Q_sector"):
        (tmp_path / f"model_{kind}_2026-09-08.csv").write_text("x\n")

    items = discover_snapshots(tmp_path)

    assert [item.kind for item in items] == ["CQBtw_Q", "CQBtw_Q_sector", "Q"]


def test_catalog_rejects_csv_without_filename_date(tmp_path):
    (tmp_path / "model_Q_latest.csv").write_text("x\n")

    with pytest.raises(ValueError, match="no YYYY-MM-DD date"):
        discover_snapshots(tmp_path)


def test_duplicate_company_key_is_a_hard_error():
    frame = pd.DataFrame([
        {"Code": "A", "FY": "2026Q4", "CQBtw": 1, "EPS_Est": 2.0},
        {"Code": "A", "FY": "2026Q4", "CQBtw": 1, "EPS_Est": 2.1},
    ])

    with pytest.raises(SchemaError, match="duplicate"):
        validate_company_snapshot(frame)


def test_company_required_fields_and_unusable_estimates_are_rejected():
    frame = pd.DataFrame([
        {"Code": "A", "FY": "2026Q4", "CQBtw": 1, "EPS_Est": 2.0},
        {"Code": None, "FY": "2026Q4", "CQBtw": 2, "EPS_Est": 1.5},
        {"Code": "C", "FY": "", "CQBtw": 3, "EPS_Est": 1.0},
        {"Code": "D", "FY": "2026Q4", "CQBtw": 4, "EPS_Est": "not-a-number"},
    ])

    result = validate_company_snapshot(frame)

    assert result.valid_rows["Code"].tolist() == ["A"]
    assert result.rejected_rows.index.tolist() == [1, 2, 3]
    assert {issue["field"] for issue in result.issues} == {"Code", "FY", "EPS_Est"}


def test_duplicate_sector_key_is_a_hard_error():
    frame = pd.DataFrame([
        {"Sector": "Tech", "FY": "2026Q4", "CQBtw": 1, "earning_total": 10.0},
        {"Sector": "Tech", "FY": "2026Q4", "CQBtw": 1, "earning_total": 12.0},
    ])

    with pytest.raises(SchemaError, match="duplicate"):
        validate_sector_snapshot(frame)


def test_metadata_provenance_lists_required_files_and_hashes(tmp_path):
    filenames = [
        "metadata_Q.txt",
        "metadata_CQBtw_Q.txt",
        "metadata_CQBtw_Q_sector.txt",
        "metadata_total_ts.txt",
    ]
    for filename in filenames:
        (tmp_path / filename).write_text(filename, encoding="utf-8")

    provenance = metadata_provenance(tmp_path)

    assert [item["path"].name for item in provenance] == filenames
    assert [item["sha256"] for item in provenance] == [
        sha256_file(tmp_path / filename) for filename in filenames
    ]


def test_metadata_provenance_reports_missing_files(tmp_path):
    (tmp_path / "metadata_Q.txt").write_text("present", encoding="utf-8")

    with pytest.raises(SchemaError, match="metadata_CQBtw_Q.txt"):
        metadata_provenance(tmp_path)

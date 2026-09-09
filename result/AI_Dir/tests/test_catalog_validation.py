from datetime import date

from consensus_context.catalog import discover_snapshots, sha256_file
import pytest


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

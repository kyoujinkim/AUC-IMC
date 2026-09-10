from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest


OLD_DATE = date(2026, 7, 31)
NEW_DATE = date(2026, 9, 8)


def _company_rows(as_of: date) -> list[dict]:
    rows = []
    is_new = as_of == NEW_DATE
    for sector, industry, sector_name, sign in (
        ("45", "4510", "정보기술", 1.0),
        ("05", "0510", "선행제로섹터", -1.0),
    ):
        for horizon in range(4):
            for company_index in range(2):
                old_eps = 10.0 + horizon + company_index
                revision = sign * (0.2 + 0.05 * horizon + 0.01 * company_index)
                rows.append(
                    {
                        "Code": f"{sector}-{company_index}",
                        "name": f"회사-{sector}-{company_index}",
                        "FY": f"FY{horizon}",
                        "CQBtw": horizon,
                        "LSector": sector,
                        "LSector_name": sector_name,
                        "Sector": industry,
                        "Sector_name": f"산업-{industry}",
                        "EPS_Est": old_eps + revision if is_new else old_eps,
                        "EPS_G": 0.08 + sign * 0.01 if is_new else 0.08,
                        "shares": 100.0 + 10.0 * company_index,
                        "model": "fixture-v1",
                    }
                )
    return rows


def _sector_rows(as_of: date) -> list[dict]:
    rows = []
    is_new = as_of == NEW_DATE
    for sector, sector_name, sign in (
        ("00", "시장전체", 1.0),
        ("45", "정보기술", 1.0),
        ("05", "선행제로섹터", -1.0),
    ):
        for horizon in range(4):
            old_total = 1000.0 + 100.0 * horizon
            rows.append(
                {
                    "Sector": sector,
                    "Sector_name": sector_name,
                    "FY": f"FY{horizon}",
                    "CQBtw": horizon,
                    "earning_total": (
                        old_total * (1.0 + sign * (0.01 + 0.005 * horizon))
                        if is_new
                        else old_total
                    ),
                    "earning_G": 0.07 + sign * 0.01 if is_new else 0.07,
                }
            )
    return rows


@pytest.fixture
def minimal_snapshot_layout(tmp_path: Path) -> dict:
    data_root = tmp_path / "data"
    metadata_root = tmp_path / "metadata"
    data_root.mkdir()
    metadata_root.mkdir()

    for as_of in (OLD_DATE, NEW_DATE):
        pd.DataFrame(_company_rows(as_of)).to_csv(
            data_root / f"fixture_Q_{as_of.isoformat()}.csv", index=False
        )
        pd.DataFrame(_sector_rows(as_of)).to_csv(
            data_root / f"fixture_CQBtw_Q_sector_{as_of.isoformat()}.csv", index=False
        )
        # Current aggregate files are present in the real layout but are not
        # the company panel (they have no Code or EPS_Est columns).
        pd.DataFrame(_sector_rows(as_of)).to_csv(
            data_root / f"fixture_CQBtw_Q_{as_of.isoformat()}.csv", index=False
        )

    # The supplied U.S. directory also contains this undated auxiliary series.
    # It is not one of the requested point-in-time snapshot families.
    (data_root / "total_ts.csv").write_text(
        "date,value\n2026-09-08,1.0\n", encoding="utf-8"
    )

    for filename in (
        "metadata_Q.txt",
        "metadata_CQBtw_Q.txt",
        "metadata_CQBtw_Q_sector.txt",
        "metadata_total_ts.txt",
    ):
        (metadata_root / filename).write_text(
            f"self-contained fixture: {filename}\n", encoding="utf-8"
        )

    selected_snapshots = sorted(data_root.glob("fixture_Q_*.csv")) + sorted(
        data_root.glob("fixture_CQBtw_Q_sector_*.csv")
    )
    metadata_paths = sorted(metadata_root.glob("*.txt"))
    return {
        "data_root": data_root,
        "metadata_root": metadata_root,
        "old_date": OLD_DATE,
        "new_date": NEW_DATE,
        "provenance_paths": selected_snapshots + metadata_paths,
        "source_paths": sorted(data_root.glob("*.csv")) + metadata_paths,
    }


@pytest.fixture
def minimal_snapshot_args(minimal_snapshot_layout: dict) -> list[str]:
    return [
        "--data-root",
        str(minimal_snapshot_layout["data_root"]),
        "--metadata-root",
        str(minimal_snapshot_layout["metadata_root"]),
        "--old-date",
        minimal_snapshot_layout["old_date"].isoformat(),
        "--new-date",
        minimal_snapshot_layout["new_date"].isoformat(),
    ]

import pandas as pd
import pytest

from consensus_context.alignment import (
    align_company_snapshots,
    align_sector_snapshots,
)
from consensus_context.features import (
    build_sector_features,
    revision_breadth,
    weighted_forward_revision,
)


def test_forward_level_excludes_current_quarter():
    revisions = {0: 0.90, 1: 0.10, 2: 0.04, 3: -0.02}
    assert weighted_forward_revision(revisions) == pytest.approx(0.055)


def test_forward_level_is_missing_when_a_required_horizon_is_missing():
    assert weighted_forward_revision({1: 0.10, 2: 0.04}) is None


def test_breadth_ignores_changes_inside_tolerance():
    result = revision_breadth([0.002, 0.0005, -0.003], tolerance=0.001)
    assert result == {"up": 1 / 3, "down": 1 / 3, "net": 0.0}


def test_company_alignment_accounts_for_matches_entrants_and_exits():
    old = pd.DataFrame(
        [
            {"Code": "A", "FY": "2026Q4", "CQBtw": 1, "LSector": "4510", "EPS_Est": 10.0},
            {"Code": "B", "FY": "2026Q4", "CQBtw": 1, "LSector": "4510", "EPS_Est": 20.0},
        ]
    )
    new = pd.DataFrame(
        [
            {"Code": "A", "FY": "2026Q4", "CQBtw": 1, "LSector": "4510", "EPS_Est": 11.0},
            {"Code": "C", "FY": "2026Q4", "CQBtw": 1, "LSector": "4510", "EPS_Est": 30.0},
        ]
    )

    panel = align_company_snapshots(old, new)

    assert panel["Code"].tolist() == ["A"]
    assert panel.loc[0, "old_EPS_Est"] == 10.0
    assert panel.loc[0, "new_EPS_Est"] == 11.0
    assert panel.loc[0, ["old_count", "new_count", "matched_count", "entrant_count", "exit_count"]].to_dict() == {
        "old_count": 2,
        "new_count": 2,
        "matched_count": 1,
        "entrant_count": 1,
        "exit_count": 1,
    }
    assert panel.loc[0, "matched_coverage"] == pytest.approx(0.5)


def test_sector_alignment_uses_the_sector_canonical_key():
    old = pd.DataFrame(
        [
            {"Sector": "4510", "FY": "2026Q4", "CQBtw": 1, "earning_total": 100.0},
            {"Sector": "4520", "FY": "2026Q4", "CQBtw": 1, "earning_total": 90.0},
        ]
    )
    new = pd.DataFrame(
        [
            {"Sector": "4510", "FY": "2026Q4", "CQBtw": 1, "earning_total": 110.0},
            {"Sector": "4530", "FY": "2026Q4", "CQBtw": 1, "earning_total": 80.0},
        ]
    )

    panel = align_sector_snapshots(old, new)

    assert panel["Sector"].tolist() == ["4510"]
    assert panel.loc[0, "old_earning_total"] == 100.0
    assert panel.loc[0, "new_earning_total"] == 110.0
    assert panel.loc[0, "entrant_count"] == 1
    assert panel.loc[0, "exit_count"] == 1


def test_sector_features_flag_denominator_instability_and_never_decide_direction():
    company_panel = pd.DataFrame(
        [
            {
                "Code": f"C{i:02d}",
                "FY": f"FY{h}",
                "CQBtw": h,
                "top_sector": "45",
                "old_EPS_Est": 10.0,
                "new_EPS_Est": 10.1,
                "old_count": 10,
                "new_count": 10,
                "matched_count": 10,
                "entrant_count": 0,
                "exit_count": 0,
                "matched_coverage": 1.0,
            }
            for h in range(4)
            for i in range(10)
        ]
    )
    sector_panel = pd.DataFrame(
        [
            {
                "Sector": "45",
                "FY": f"FY{h}",
                "CQBtw": h,
                "old_earning_total": -1.0 if h == 1 else 100.0,
                "new_earning_total": 1.0 if h == 1 else 101.0,
                "old_earning_G": 0.08,
                "new_earning_G": 0.09,
            }
            for h in range(4)
        ]
    )

    result = build_sector_features(company_panel, sector_panel)

    assert len(result) == 1
    sector = result[0]
    assert sector["revisions"][1] == pytest.approx(2.0)
    assert sector["quality_flags"]["denominator_instability"] is True
    assert "decision" not in sector
    assert "direction" not in sector


def test_sector_features_report_coverage_conflict_model_change_and_component_ranks():
    company_rows = []
    for sector, old_model, new_model, new_eps in (
        ("45", "legacy", "new-model", 9.8),
        ("50", "stable", "stable", 10.0),
    ):
        for horizon in range(4):
            for company in range(8):
                company_rows.append(
                    {
                        "Code": f"{sector}-{company}",
                        "FY": f"FY{horizon}",
                        "CQBtw": horizon,
                        "top_sector": sector,
                        "old_EPS_Est": 10.0,
                        "new_EPS_Est": new_eps,
                        "old_EPS_G": 0.05,
                        "new_EPS_G": 0.06,
                        "old_model": old_model,
                        "new_model": new_model,
                        "old_count": 20,
                        "new_count": 20,
                        "matched_count": 8,
                        "entrant_count": 12,
                        "exit_count": 12,
                        "matched_coverage": 0.4,
                    }
                )
    company_panel = pd.DataFrame(company_rows)
    sector_panel = pd.DataFrame(
        [
            {
                "Sector": sector,
                "FY": f"FY{horizon}",
                "CQBtw": horizon,
                "old_earning_total": 100.0,
                "new_earning_total": 105.0 if sector == "45" else 100.0,
                "old_earning_G": 0.05,
                "new_earning_G": 0.07 if sector == "45" else 0.05,
            }
            for sector in ("45", "50")
            for horizon in range(4)
        ]
    )

    result = {row["sector"]: row for row in build_sector_features(company_panel, sector_panel)}

    flagged = result["45"]
    assert flagged["quality_flags"]["insufficient_coverage"] is True
    assert flagged["quality_flags"]["insufficient_matched_companies"] is True
    assert flagged["quality_flags"]["aggregate_median_conflict"] is True
    assert flagged["quality_flags"]["model_composition_change"] is True
    assert set(flagged["component_ranks"]) == {"forward", "breadth", "slope", "growth"}
    assert flagged["research_prior"] is not None
    assert all(0.0 <= value <= 1.0 for value in flagged["component_ranks"].values())


def test_fewer_than_ten_matches_trips_the_combined_coverage_gate():
    company_panel = pd.DataFrame(
        [
            {
                "Code": f"C{company}",
                "FY": f"FY{horizon}",
                "CQBtw": horizon,
                "top_sector": "45",
                "old_EPS_Est": 10.0,
                "new_EPS_Est": 10.1,
                "old_count": 8,
                "new_count": 8,
                "matched_count": 8,
                "entrant_count": 0,
                "exit_count": 0,
                "matched_coverage": 1.0,
            }
            for horizon in range(1, 4)
            for company in range(8)
        ]
    )

    result = build_sector_features(company_panel, pd.DataFrame())[0]

    assert result["quality_flags"]["insufficient_coverage"] is True
    assert result["quality_flags"]["insufficient_matched_companies"] is True

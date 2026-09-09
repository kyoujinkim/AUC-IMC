import pandas as pd
import pytest

from consensus_context import features
from consensus_context.alignment import (
    align_company_snapshots,
    align_sector_snapshots,
)
from consensus_context.contribution import (
    attribute_contributors,
    truncate_contributors,
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


def test_sector_totals_exclude_rows_without_a_finite_old_new_pair():
    sector_panel = pd.DataFrame(
        [
            {
                "Sector": "4510",
                "FY": "FY1",
                "CQBtw": 1,
                "old_earning_total": 100.0,
                "new_earning_total": 110.0,
            },
            {
                "Sector": "4520",
                "FY": "FY1",
                "CQBtw": 1,
                "old_earning_total": 100.0,
                "new_earning_total": float("nan"),
            },
        ]
    )

    result = build_sector_features(pd.DataFrame(), sector_panel)[0]

    assert result["revisions"][1] == pytest.approx(0.10)


def test_coverage_only_sector_is_emitted_with_unavailable_signals_and_flags():
    company_panel = pd.DataFrame()
    company_panel.attrs["coverage"] = [
        {
            "top_sector": "45",
            "FY": "FY1",
            "CQBtw": 1,
            "old_count": 12,
            "new_count": 4,
            "matched_count": 0,
            "entrant_count": 4,
            "exit_count": 12,
            "matched_coverage": 0.0,
        }
    ]

    result = build_sector_features(company_panel, pd.DataFrame())

    assert len(result) == 1
    sector = result[0]
    assert sector["sector"] == "45"
    assert sector["revisions"] == {0: None, 1: None, 2: None, 3: None}
    assert sector["forward_revision"] is None
    assert sector["coverage"][1]["entrant_count"] == 4
    assert sector["quality_flags"]["insufficient_coverage"] is True
    assert sector["quality_flags"]["insufficient_matched_companies"] is True


def test_company_sector_migration_is_an_exit_and_entrant_but_remains_matched():
    old = pd.DataFrame(
        [
            {
                "Code": "A",
                "FY": "2026Q4",
                "CQBtw": 1,
                "LSector": "4510",
                "EPS_Est": 10.0,
            }
        ]
    )
    new = pd.DataFrame(
        [
            {
                "Code": "A",
                "FY": "2026Q4",
                "CQBtw": 1,
                "LSector": "5010",
                "EPS_Est": 11.0,
            }
        ]
    )

    panel = align_company_snapshots(old, new)
    coverage = {row["top_sector"]: row for row in panel.attrs["coverage"]}

    assert panel["Code"].tolist() == ["A"]
    assert coverage["45"]["exit_count"] == 1
    assert coverage["45"]["matched_count"] == 0
    assert coverage["50"]["entrant_count"] == 1
    assert coverage["50"]["matched_count"] == 0
    assert coverage["50"]["matched_coverage"] == 0.0


def test_model_composition_change_is_evaluated_within_each_horizon():
    company_panel = pd.DataFrame(
        [
            {
                "Code": "A",
                "FY": "FY1",
                "CQBtw": 1,
                "top_sector": "45",
                "old_EPS_Est": 10.0,
                "new_EPS_Est": 10.1,
                "old_model": "legacy",
                "new_model": "modern",
            },
            {
                "Code": "B",
                "FY": "FY2",
                "CQBtw": 2,
                "top_sector": "45",
                "old_EPS_Est": 10.0,
                "new_EPS_Est": 10.1,
                "old_model": "modern",
                "new_model": "legacy",
            },
        ]
    )

    result = build_sector_features(company_panel, pd.DataFrame())[0]

    assert result["quality_flags"]["model_composition_change"] is True


def test_quality_gate_constants_are_named_and_match_the_research_policy():
    assert features.MIN_MATCHED_COVERAGE == 0.50
    assert features.MIN_MATCHED_COMPANIES == 10
    assert features.MIN_FLAGGED_FORWARD_HORIZONS == 2


def test_truncation_keeps_both_tails_and_discloses_omissions():
    rows = [{"id": str(i), "contribution": float(i - 15)} for i in range(31)]

    result = truncate_contributors(rows, top_n=10)

    assert len(result["positive"]) == 10
    assert len(result["negative"]) == 10
    assert result["omitted_count"] == 11
    assert result["all_contributors_total"] == sum(
        row["contribution"] for row in rows
    )
    assert result["omitted_signed_total"] == pytest.approx(0.0)
    assert result["gross_total"] == pytest.approx(240.0)
    assert result["total_count"] == 31


def test_truncation_uses_entity_id_as_the_stable_tie_break():
    rows = [
        {"id": "B", "contribution": 3.0},
        {"id": "A", "contribution": 3.0},
        {"id": "D", "contribution": -2.0},
        {"id": "C", "contribution": -2.0},
    ]

    result = truncate_contributors(rows, top_n=2)

    assert [row["id"] for row in result["positive"]] == ["A", "B"]
    assert [row["id"] for row in result["negative"]] == ["C", "D"]


def test_attribution_uses_prior_shares_and_preserves_identity_and_horizon():
    panel = pd.DataFrame(
        [
            {
                "Code": "A",
                "FY": "2027Q1",
                "CQBtw": 1,
                "top_sector": "45",
                "new_Sector": "4510",
                "new_name": "Alpha",
                "old_EPS_Est": 2.0,
                "new_EPS_Est": 2.5,
                "old_shares": 100.0,
                "new_shares": 1_000.0,
            }
        ]
    )

    result = attribute_contributors(panel)
    horizon = result["45"][1]
    company = horizon["companies"]["positive"][0]

    assert company == {
        "id": "A",
        "name": "Alpha",
        "sector": "45",
        "industry": "4510",
        "horizon": 1,
        "fy": "2027Q1",
        "contribution": pytest.approx(50.0),
        "absolute_contribution_share": pytest.approx(1.0),
    }
    assert horizon["all_contributors_total"] == pytest.approx(50.0)


def test_attribution_excludes_non_finite_inputs_and_surfaces_the_count():
    panel = pd.DataFrame(
        [
            {
                "Code": code,
                "FY": "FY1",
                "CQBtw": 1,
                "top_sector": "45",
                "new_Sector": "4510",
                "old_EPS_Est": old_eps,
                "new_EPS_Est": new_eps,
                "old_shares": old_shares,
            }
            for code, old_eps, new_eps, old_shares in (
                ("valid", 1.0, 2.0, 10.0),
                ("old-eps", float("nan"), 2.0, 10.0),
                ("new-eps", 1.0, float("inf"), 10.0),
                ("shares", 1.0, 2.0, float("nan")),
            )
        ]
    )

    horizon = attribute_contributors(panel)["45"][1]

    assert horizon["excluded_count"] == 3
    assert horizon["total_count"] == 1
    assert horizon["all_contributors_total"] == pytest.approx(10.0)


def test_attribution_aggregates_industries_and_companies_with_gross_shares():
    panel = pd.DataFrame(
        [
            {
                "Code": code,
                "FY": "FY2",
                "CQBtw": 2,
                "top_sector": "45",
                "new_Sector": industry,
                "old_EPS_Est": 10.0,
                "new_EPS_Est": 10.0 + contribution,
                "old_shares": 1.0,
            }
            for code, industry, contribution in (
                ("A", "4510", 4.0),
                ("B", "4510", -1.0),
                ("C", "4520", -2.0),
            )
        ]
    )

    horizon = attribute_contributors(panel)["45"][2]
    industries = {
        row["id"]: row
        for tail in ("positive", "negative")
        for row in horizon["industries"][tail]
    }
    companies = {
        row["id"]: row
        for tail in ("positive", "negative")
        for row in horizon["companies"][tail]
    }

    assert horizon["all_contributors_total"] == pytest.approx(1.0)
    assert horizon["gross_total"] == pytest.approx(7.0)
    assert companies["A"]["absolute_contribution_share"] == pytest.approx(4 / 7)
    assert industries["4510"]["contribution"] == pytest.approx(3.0)
    assert industries["4510"]["absolute_contribution_share"] == pytest.approx(3 / 7)
    assert industries["4520"]["contribution"] == pytest.approx(-2.0)


def test_attribution_calculates_concentration_and_reconciliation():
    panel = pd.DataFrame(
        [
            {
                "Code": code,
                "FY": "FY1",
                "CQBtw": 1,
                "top_sector": "45",
                "new_Sector": "4510",
                "old_EPS_Est": 10.0,
                "new_EPS_Est": 10.0 + contribution,
                "old_shares": 1.0,
                "old_earning_total": 100.0,
                "new_earning_total": 112.0,
            }
            for code, contribution in (
                ("A", 4.0),
                ("B", 3.0),
                ("C", -2.0),
                ("D", 1.0),
                ("E", -1.0),
                ("F", 1.0),
            )
        ]
    )

    horizon = attribute_contributors(panel)["45"][1]

    assert horizon["top_three_gross_share"] == pytest.approx(9 / 12)
    assert horizon["top_five_gross_share"] == pytest.approx(11 / 12)
    assert horizon["contribution_hhi"] == pytest.approx(32 / 144)
    assert horizon["supplied_earning_total_change"] == pytest.approx(12.0)
    assert horizon["bottom_up_total"] == pytest.approx(6.0)
    assert horizon["reconciliation_residual"] == pytest.approx(6.0)

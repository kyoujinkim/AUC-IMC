import json
import math
from copy import deepcopy
from datetime import date, datetime, timezone
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from consensus_context.contracts import (
    build_research_context,
    ensure_user_context,
    initial_user_context,
    validate_analysis_result,
)
from consensus_context.serialization import write_pretty_json


SCHEMA_DIR = Path(__file__).parents[1] / "schemas"


def _schema(name):
    return json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))


@pytest.fixture
def valid_research():
    return build_research_context(
        run_id="research-2026-09-08",
        generated_at=datetime(2026, 9, 8, 12, 30, tzinfo=timezone.utc),
        old_date=date(2026, 7, 31),
        new_date=date(2026, 9, 8),
        provenance=[
            {
                "kind": "CQBtw_Q_sector",
                "path": Path("us/CQBtw_Q_sector_2026-09-08.csv"),
                "sha256": "a" * 64,
            }
        ],
        methodology={"forward_weights": {"1": 0.45, "2": 0.35, "3": 0.20}},
        sectors=[
            {
                "sector": "45",
                "sector_name": "정보기술",
                "signals": {"forward_revision": 0.05},
            }
        ],
        validation_summary={"rejected_rows": 0, "issues": []},
    )


@pytest.fixture
def valid_analysis():
    return {
        "schema_version": "1.0",
        "research_run_id": "research-2026-09-08",
        "user_context_version": 1,
        "long_sectors": [
            {
                "sector": "45",
                "sector_name": "정보기술",
                "confidence": "Medium",
                "provisional": False,
                "thesis": "Forward estimates improved with broad support.",
                "quantitative_support": [
                    {
                        "metric": "forward_revision",
                        "value": 0.05,
                        "research_json_path": "/sectors/0/signals/forward_revision",
                    }
                ],
                "drivers": [
                    {
                        "claim": "Guidance supports the estimate change.",
                        "status": "Confirmed",
                        "evidence_ids": ["evidence-1"],
                    }
                ],
                "evidence_ids": ["evidence-1"],
                "counter_evidence": ["Demand may normalize."],
                "impact": "Supports a positive 1-3 month view.",
                "risks": ["Guidance could reverse."],
                "invalidation_conditions": ["Forward revision turns negative."],
            }
        ],
        "short_sectors": [],
        "neutral_sectors": [],
        "watch_sectors": [],
        "evidence_ledger": [
            {
                "evidence_id": "evidence-1",
                "status": "Confirmed",
                "source": "Issuer release",
                "publication_date": "2026-08-15",
                "claim": "Guidance increased.",
            }
        ],
        "data_requests": [],
    }


def test_pretty_json_is_utf8_indented_and_newline_terminated(tmp_path):
    path = tmp_path / "context.json"

    write_pretty_json(path, {"sector_name": "정보기술", "value": 1.0})

    payload = path.read_bytes()
    assert payload.endswith(b"\n")
    text = payload.decode("utf-8")
    assert '  "sector_name": "정보기술"' in text
    assert "\\u" not in text
    assert json.loads(text) == {"sector_name": "정보기술", "value": 1.0}


def test_nonfinite_value_is_rejected_without_replacing_existing_file(tmp_path):
    path = tmp_path / "context.json"
    original = b'{"preserve": true}\n'
    path.write_bytes(original)

    with pytest.raises(ValueError):
        write_pretty_json(path, {"value": math.nan})

    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_research_context_has_required_fact_fields_and_validates(valid_research):
    assert valid_research["schema_version"] == "1.0"
    assert valid_research["run_id"] == "research-2026-09-08"
    assert valid_research["generated_at"] == "2026-09-08T12:30:00Z"
    assert valid_research["snapshot_dates"] == {
        "old": "2026-07-31",
        "new": "2026-09-08",
    }
    assert valid_research["provenance"][0]["path"] == (
        "us/CQBtw_Q_sector_2026-09-08.csv"
    )
    assert list(Draft202012Validator(_schema("research-context.schema.json")).iter_errors(valid_research)) == []


@pytest.mark.parametrize("forbidden", ["decision", "drivers", "causal_claim"])
def test_research_context_rejects_decision_and_causal_fields(forbidden):
    with pytest.raises(ValueError, match=forbidden):
        build_research_context(
            run_id="run-1",
            generated_at="2026-09-08T12:30:00Z",
            old_date="2026-07-31",
            new_date="2026-09-08",
            provenance=[],
            methodology={forbidden: "must not leak"},
            sectors=[],
            validation_summary={},
        )


def test_research_context_rejects_non_iso_snapshot_dates():
    with pytest.raises(ValueError, match="snapshot_dates"):
        build_research_context(
            run_id="run-1",
            generated_at="2026-09-08T12:30:00Z",
            old_date="July 31",
            new_date="2026-09-08",
            provenance=[],
            methodology={},
            sectors=[],
            validation_summary={},
        )


def test_research_context_rejects_nested_nonfinite_values():
    with pytest.raises(ValueError, match="non-finite"):
        build_research_context(
            run_id="run-1",
            generated_at="2026-09-08T12:30:00Z",
            old_date="2026-07-31",
            new_date="2026-09-08",
            provenance=[],
            methodology={},
            sectors=[{"sector": "45", "signals": {"forward_revision": math.inf}}],
            validation_summary={},
        )


def test_user_context_is_created_once_and_existing_bytes_are_preserved(tmp_path):
    path = tmp_path / "user_context.json"

    created = ensure_user_context(path, "research-run-1")
    assert created == initial_user_context("research-run-1")
    Draft202012Validator(_schema("user-context.schema.json")).validate(created)

    custom_bytes = (
        b'{\n  "schema_version": "1.0",\n  "user_context_version": 7,\n'
        b'  "research_run_id": "research-run-1",\n  "preferences": {},\n'
        b'  "notes": [],\n  "evidence_inputs": [],\n  "request_responses": {},\n'
        b'  "desk_extension": {"owner": "quant"}\n}\n'
    )
    path.write_bytes(custom_bytes)

    loaded = ensure_user_context(path, "a-new-research-run")

    assert loaded["desk_extension"] == {"owner": "quant"}
    assert path.read_bytes() == custom_bytes


def test_analysis_schema_and_traceability_accept_valid_result(
    valid_analysis, valid_research
):
    assert validate_analysis_result(valid_analysis, valid_research) == []
    Draft202012Validator(_schema("analysis-result.schema.json")).validate(valid_analysis)


def test_analysis_rejects_untraceable_or_mismatched_quantitative_support(
    valid_analysis, valid_research
):
    mismatched = deepcopy(valid_analysis)
    mismatched["long_sectors"][0]["quantitative_support"][0]["value"] = 999
    invalid_path = deepcopy(valid_analysis)
    invalid_path["long_sectors"][0]["quantitative_support"][0][
        "research_json_path"
    ] = "/sectors/9/signals/forward_revision"

    assert any(
        "research_json_path" in error
        for error in validate_analysis_result(mismatched, valid_research)
    )
    assert any(
        "research_json_path" in error
        for error in validate_analysis_result(invalid_path, valid_research)
    )


def test_analysis_allows_an_explicit_numeric_tolerance(valid_analysis, valid_research):
    support = valid_analysis["long_sectors"][0]["quantitative_support"][0]
    support["value"] = 0.0505
    support["numeric_tolerance"] = 0.001

    assert validate_analysis_result(valid_analysis, valid_research) == []


def test_analysis_rejects_unknown_evidence_and_missing_risk_contracts(
    valid_analysis, valid_research
):
    invalid = deepcopy(valid_analysis)
    decision = invalid["long_sectors"][0]
    decision["evidence_ids"] = ["missing-evidence"]
    decision["risks"] = []
    decision["invalidation_conditions"] = []

    errors = validate_analysis_result(invalid, valid_research)

    assert any("missing-evidence" in error for error in errors)
    assert any("risks" in error for error in errors)
    assert any("invalidation_conditions" in error for error in errors)


def test_analysis_rejects_unknown_driver_evidence_and_wrong_research_run(
    valid_analysis, valid_research
):
    invalid = deepcopy(valid_analysis)
    invalid["research_run_id"] = "different-run"
    invalid["long_sectors"][0]["drivers"][0]["evidence_ids"] = [
        "missing-driver-evidence"
    ]

    errors = validate_analysis_result(invalid, valid_research)

    assert any("research_run_id" in error for error in errors)
    assert any("missing-driver-evidence" in error for error in errors)


def test_analysis_rejects_recursive_sizing_fields(valid_analysis, valid_research):
    for forbidden in ("weight", "allocation", "position_size", "leverage"):
        invalid = deepcopy(valid_analysis)
        invalid["long_sectors"][0]["drivers"][0][forbidden] = 0.2

        errors = validate_analysis_result(invalid, valid_research)

        assert any(forbidden in error for error in errors)


def test_analysis_rejects_more_than_three_per_side_and_duplicate_sectors(
    valid_analysis, valid_research
):
    too_many = deepcopy(valid_analysis)
    template = too_many["long_sectors"][0]
    too_many["long_sectors"] = []
    for index in range(4):
        decision = deepcopy(template)
        decision["sector"] = str(index)
        too_many["long_sectors"].append(decision)

    duplicate = deepcopy(valid_analysis)
    duplicate["short_sectors"] = [deepcopy(duplicate["long_sectors"][0])]

    assert any("long_sectors" in error for error in validate_analysis_result(too_many, valid_research))
    assert any("unique" in error for error in validate_analysis_result(duplicate, valid_research))

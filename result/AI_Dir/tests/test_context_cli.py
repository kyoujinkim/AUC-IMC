from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from consensus_context import build_context
from consensus_context.catalog import sha256_file
from consensus_context.cli import main


def _walk_keys(value):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield key.casefold()
            yield from _walk_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_keys(nested)


def test_cli_emits_pretty_fact_context_with_provenance(
    minimal_snapshot_args, minimal_snapshot_layout, tmp_path
):
    source_hashes = {
        path: sha256_file(path) for path in minimal_snapshot_layout["source_paths"]
    }
    provenance_hashes = {
        path: sha256_file(path)
        for path in minimal_snapshot_layout["provenance_paths"]
    }
    output = tmp_path / "run"

    assert main([*minimal_snapshot_args, "--output", str(output)]) == 0

    research_path = output / "research_context.json"
    research_bytes = research_path.read_bytes()
    research = json.loads(research_bytes.decode("utf-8"))
    manifest = json.loads((output / "build_manifest.json").read_text(encoding="utf-8"))

    assert research_bytes.endswith(b"\n")
    assert b'  "schema_version"' in research_bytes
    assert research["snapshot_dates"] == {"old": "2026-07-31", "new": "2026-09-08"}
    assert [sector["sector"] for sector in research["sectors"]] == ["05", "45"]
    by_sector = {sector["sector"]: sector for sector in research["sectors"]}
    assert by_sector["45"]["forward_revision"] > 0
    assert by_sector["05"]["forward_revision"] < 0
    horizon_one = by_sector["45"]["contributions"]["1"]
    assert horizon_one["total_count"] == 2
    assert horizon_one["supplied_earning_total_change"] == pytest.approx(16.5)
    assert horizon_one["reconciliation_residual"] == pytest.approx(-37.1)
    assert (output / "user_context.json").is_file()
    assert {item["sha256"] for item in research["provenance"]} == set(
        provenance_hashes.values()
    )
    assert manifest["research_context_sha256"] == sha256_file(research_path)
    assert manifest["source_sha256"] == {
        path.as_posix(): digest for path, digest in provenance_hashes.items()
    }
    assert not ({"decision", "decisions", "long", "short"} & set(_walk_keys(research)))
    assert all(sha256_file(path) == digest for path, digest in source_hashes.items())


def test_build_context_preserves_edited_user_context_bytes_on_rerun(
    minimal_snapshot_layout, tmp_path
):
    output = tmp_path / "run"
    kwargs = {
        "data_root": minimal_snapshot_layout["data_root"],
        "metadata_root": minimal_snapshot_layout["metadata_root"],
        "old_date": minimal_snapshot_layout["old_date"],
        "new_date": minimal_snapshot_layout["new_date"],
        "output_dir": output,
        "top_n": 1,
    }

    first_path = build_context(**kwargs)
    assert first_path == output / "research_context.json"
    user_path = output / "user_context.json"
    user = json.loads(user_path.read_text(encoding="utf-8"))
    user["user_context_version"] = 2
    user["notes"] = ["수동 편집은 재생성 후에도 유지"]
    user["request_responses"] = {
        "request-1": {"status": "answered", "response": "확인 완료"}
    }
    manual_bytes = json.dumps(user, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    user_path.write_bytes(manual_bytes)

    second_path = build_context(**kwargs)

    assert second_path == first_path
    assert user_path.read_bytes() == manual_bytes
    research = json.loads(second_path.read_text(encoding="utf-8"))
    assert research["sectors"][0]["contributions"]["1"]["companies"][
        "total_count"
    ] == 2


def test_main_rejects_non_positive_top_contributor_limit(
    minimal_snapshot_args, tmp_path, capsys
):
    result = main(
        [
            *minimal_snapshot_args,
            "--output",
            str(tmp_path / "run"),
            "--top-contributors",
            "0",
        ]
    )

    assert result == 2
    assert "positive integer" in capsys.readouterr().err


def test_python_module_entry_point_runs_without_duplicate_import_warning(
    minimal_snapshot_args, tmp_path
):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "consensus_context.cli",
            *minimal_snapshot_args,
            "--output",
            str(tmp_path / "module-run"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "RuntimeWarning" not in completed.stderr


def test_build_context_rejects_missing_requested_snapshot(
    minimal_snapshot_layout, tmp_path
):
    missing_date = minimal_snapshot_layout["new_date"].replace(day=9)

    try:
        build_context(
            data_root=minimal_snapshot_layout["data_root"],
            metadata_root=minimal_snapshot_layout["metadata_root"],
            old_date=minimal_snapshot_layout["old_date"],
            new_date=missing_date,
            output_dir=tmp_path / "run",
        )
    except ValueError as exc:
        assert "2026-09-09" in str(exc)
    else:
        raise AssertionError("missing requested snapshot pair was accepted")

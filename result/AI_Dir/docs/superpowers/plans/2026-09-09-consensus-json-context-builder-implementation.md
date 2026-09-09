# Consensus JSON Context Builder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a thin deterministic Python layer that preserves source CSV files and emits a compact, human-readable `research_context.json` plus a safe editable `user_context.json` for LLM-native sector research.

**Architecture:** Python performs only schema validation, point-in-time alignment, canonical calculations, contributor attribution, provenance, and JSON validation. It neither selects Long/Short sectors nor writes causal explanations. The LLM consumes the generated context and separate user context, then writes `analysis_result.json` and `report.md` under the companion skill plan.

**Tech Stack:** Python 3.10+, pandas 2.x, NumPy 1.26+, pytest 8.x, standard-library argparse/dataclasses/hashlib/json/pathlib.

**Spec:** `docs/superpowers/specs/2026-09-09-consensus-investment-report-design.md`

## Global Constraints

- Preserve all source metadata and CSV files byte-for-byte; write outputs only below the requested output directory.
- Persist no SQL database, parquet store, or opaque binary cache.
- Serialize JSON as UTF-8 with `ensure_ascii=False`, `indent=2`, ISO dates, stable IDs, and no `NaN` or infinity.
- `research_context.json` contains reproducible facts only and is read-only by convention.
- `user_context.json` contains editable user material only and must survive context regeneration unchanged.
- The deterministic layer never emits Long, Short, Neutral, or Watch decisions and never writes causal prose.
- The current quarter is secondary; canonical forward level uses horizons 1–3 with weights `0.45`, `0.35`, and `0.20`.
- Contributor detail defaults to the top 10 positive and top 10 negative companies and industries per sector, with omitted counts and totals disclosed.
- No portfolio weights, leverage, order sizing, or execution instructions appear in any artifact.

## Planned File Structure

```text
result/AI_Dir/
├── pyproject.toml
├── consensus_context/
│   ├── __init__.py            # Public context-builder API
│   ├── catalog.py             # Snapshot discovery and source hashes
│   ├── validation.py          # Metadata/schema/key validation
│   ├── alignment.py           # Point-in-time comparable panels
│   ├── features.py            # Canonical revision and breadth metrics
│   ├── contribution.py        # Industry/company attribution and truncation
│   ├── contracts.py           # JSON object construction and validation
│   ├── serialization.py       # Atomic pretty JSON output
│   └── cli.py                 # build-context command
├── schemas/
│   ├── research-context.schema.json
│   ├── user-context.schema.json
│   └── analysis-result.schema.json
├── tests/
│   ├── fixtures/
│   ├── test_catalog_validation.py
│   ├── test_features_contribution.py
│   ├── test_json_contracts.py
│   └── test_context_cli.py
└── docs/
    └── consensus-json-contract.md
```

---

### Task 1: Package, Typed Configuration, and Source Catalog

**Files:**
- Create: `pyproject.toml`
- Create: `consensus_context/__init__.py`
- Create: `consensus_context/catalog.py`
- Create: `tests/test_catalog_validation.py`

**Interfaces:**
- Produces: `SnapshotFile(kind: str, as_of_date: date, path: Path, sha256: str)`.
- Produces: `discover_snapshots(data_root: Path) -> list[SnapshotFile]`.
- Produces: `sha256_file(path: Path) -> str`.

- [ ] **Step 1: Write the failing filename and hashing tests**

```python
from datetime import date
from consensus_context.catalog import discover_snapshots, sha256_file

def test_catalog_uses_filename_date_and_hash(tmp_path):
    source = tmp_path / "mixed_model_Q_2026-09-08.csv"
    source.write_text("Code,FY,CQBtw,EPS_Est\nA,2026Q4,1,2.0\n", encoding="utf-8")
    items = discover_snapshots(tmp_path)
    assert items[0].as_of_date == date(2026, 9, 8)
    assert items[0].sha256 == sha256_file(source)
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `python -m pytest tests/test_catalog_validation.py -v`
Expected: FAIL because `consensus_context.catalog` does not exist.

- [ ] **Step 3: Implement the immutable catalog**

Use a frozen dataclass, a filename regex ending in `YYYY-MM-DD.csv`, streaming SHA-256 reads, and explicit kind recognition for `Q`, `CQBtw_Q`, and `CQBtw_Q_sector`. Reject filenames without a date; never fall back to modification time.

- [ ] **Step 4: Add package metadata and CLI entry point**

```toml
[project]
name = "consensus-context"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["pandas>=2,<3", "numpy>=1.26,<3", "jsonschema>=4,<5"]

[project.scripts]
build-consensus-context = "consensus_context.cli:main"
```

- [ ] **Step 5: Run the focused test and commit**

Run: `python -m pytest tests/test_catalog_validation.py -v`
Expected: PASS.

```bash
git add result/AI_Dir/pyproject.toml result/AI_Dir/consensus_context result/AI_Dir/tests/test_catalog_validation.py
git commit -m "feat: catalog consensus snapshots"
```

### Task 2: Metadata, Schema, and Canonical-Key Validation

**Files:**
- Create: `consensus_context/validation.py`
- Modify: `tests/test_catalog_validation.py`

**Interfaces:**
- Consumes: `SnapshotFile`.
- Produces: `ValidationResult(valid_rows: DataFrame, rejected_rows: DataFrame, issues: list[dict])`.
- Produces: `validate_company_snapshot(frame: DataFrame) -> ValidationResult`.
- Produces: `validate_sector_snapshot(frame: DataFrame) -> ValidationResult`.

- [ ] **Step 1: Add failing tests for required columns and duplicate keys**

```python
import pandas as pd
import pytest
from consensus_context.validation import SchemaError, validate_company_snapshot

def test_duplicate_company_key_is_a_hard_error():
    frame = pd.DataFrame([
        {"Code": "A", "FY": "2026Q4", "CQBtw": 1, "EPS_Est": 2.0},
        {"Code": "A", "FY": "2026Q4", "CQBtw": 1, "EPS_Est": 2.1},
    ])
    with pytest.raises(SchemaError, match="duplicate"):
        validate_company_snapshot(frame)
```

- [ ] **Step 2: Run the focused test and verify failure**

Run: `python -m pytest tests/test_catalog_validation.py -v`
Expected: FAIL because validation interfaces are undefined.

- [ ] **Step 3: Implement explicit schemas and rejection accounting**

Company keys are `Code × FY × CQBtw`; sector keys are `Sector × FY × CQBtw`. Required identifier or estimate omissions enter `rejected_rows`; duplicate canonical keys raise `SchemaError`. Metadata files are read before columns are interpreted, and their paths and hashes enter provenance.

- [ ] **Step 4: Run tests and commit**

Run: `python -m pytest tests/test_catalog_validation.py -v`
Expected: PASS.

```bash
git add result/AI_Dir/consensus_context/validation.py result/AI_Dir/tests/test_catalog_validation.py
git commit -m "feat: validate consensus schemas"
```

### Task 3: Alignment, Forward Features, and Quality Flags

**Files:**
- Create: `consensus_context/alignment.py`
- Create: `consensus_context/features.py`
- Create: `tests/test_features_contribution.py`

**Interfaces:**
- Produces: `align_company_snapshots(old: DataFrame, new: DataFrame) -> DataFrame`.
- Produces: `align_sector_snapshots(old: DataFrame, new: DataFrame) -> DataFrame`.
- Produces: `build_sector_features(company_panel: DataFrame, sector_panel: DataFrame) -> list[dict]`.

- [ ] **Step 1: Write failing horizon-weight and breadth tests**

```python
import pytest
from consensus_context.features import weighted_forward_revision, revision_breadth

def test_forward_level_excludes_current_quarter():
    revisions = {0: 0.90, 1: 0.10, 2: 0.04, 3: -0.02}
    assert weighted_forward_revision(revisions) == pytest.approx(0.055)

def test_breadth_ignores_changes_inside_tolerance():
    result = revision_breadth([0.002, 0.0005, -0.003], tolerance=0.001)
    assert result == {"up": 1/3, "down": 1/3, "net": 0.0}
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_features_contribution.py -v`
Expected: FAIL because feature functions do not exist.

- [ ] **Step 3: Implement exact point-in-time joins and coverage**

Use inner joins for canonical revisions and separate left/right anti-joins for entrants and exits. For every sector and horizon retain old, new, matched, entrant, and exit counts plus matched coverage. Never convert an unmatched row into a zero revision.

- [ ] **Step 4: Implement canonical metrics**

Calculate raw and scaled revisions, horizons 1–3 weighted level, horizon 0–3 slope, up/down/net breadth with ±0.1% tolerance, mean, median, trimmed mean, growth level/change, robust normalized components, and the transparent research prior. Attach the exact quality flags from sections 6–8 of the spec.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/test_features_contribution.py -v`
Expected: PASS.

```bash
git add result/AI_Dir/consensus_context/alignment.py result/AI_Dir/consensus_context/features.py result/AI_Dir/tests/test_features_contribution.py
git commit -m "feat: calculate consensus context features"
```

### Task 4: Bounded Contributor Attribution

**Files:**
- Create: `consensus_context/contribution.py`
- Modify: `tests/test_features_contribution.py`

**Interfaces:**
- Produces: `attribute_contributors(company_panel: DataFrame, top_n: int = 10) -> dict[str, dict]`.

- [ ] **Step 1: Write the failing reconciliation and truncation test**

```python
from consensus_context.contribution import truncate_contributors

def test_truncation_keeps_both_tails_and_discloses_omissions():
    rows = [{"id": str(i), "contribution": float(i - 15)} for i in range(31)]
    result = truncate_contributors(rows, top_n=10)
    assert len(result["positive"]) == 10
    assert len(result["negative"]) == 10
    assert result["omitted_count"] == 11
    assert result["all_contributors_total"] == sum(r["contribution"] for r in rows)
```

- [ ] **Step 2: Run the test and verify failure**

Run: `python -m pytest tests/test_features_contribution.py -v`
Expected: FAIL because contributor functions do not exist.

- [ ] **Step 3: Implement profit attribution and compact detail**

Use previous-snapshot shares in `(new EPS - old EPS) × old shares`. Aggregate by company, industry, and sector; calculate gross-share concentration, top-three/top-five shares, HHI, supplied-versus-bottom-up residual, omitted count, omitted signed total, and total contributor count. Sort stable ties by canonical entity ID.

- [ ] **Step 4: Run tests and commit**

Run: `python -m pytest tests/test_features_contribution.py -v`
Expected: PASS.

```bash
git add result/AI_Dir/consensus_context/contribution.py result/AI_Dir/tests/test_features_contribution.py
git commit -m "feat: add bounded consensus attribution"
```

### Task 5: Three JSON Contracts and Safe Serialization

**Files:**
- Create: `consensus_context/contracts.py`
- Create: `consensus_context/serialization.py`
- Create: `schemas/research-context.schema.json`
- Create: `schemas/user-context.schema.json`
- Create: `schemas/analysis-result.schema.json`
- Create: `tests/test_json_contracts.py`

**Interfaces:**
- Produces: `build_research_context(...) -> dict`.
- Produces: `initial_user_context(research_run_id: str) -> dict`.
- Produces: `write_pretty_json(path: Path, value: dict) -> None`.
- Produces: `validate_analysis_result(result: dict, research: dict) -> list[str]`.

- [ ] **Step 1: Write failing serialization and separation tests**

```python
import json
import math
import pytest
from consensus_context.serialization import write_pretty_json

def test_pretty_json_is_utf8_indented_and_rejects_nan(tmp_path):
    path = tmp_path / "context.json"
    write_pretty_json(path, {"sector_name": "정보기술", "value": 1.0})
    text = path.read_text(encoding="utf-8")
    assert '  "sector_name": "정보기술"' in text
    assert "\\uC815" not in text
    with pytest.raises(ValueError):
        write_pretty_json(path, {"value": math.nan})
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_json_contracts.py -v`
Expected: FAIL because contract and serializer modules do not exist.

- [ ] **Step 3: Define closed generated schemas and extensible user schema**

Require `schema_version`, `run_id`, `generated_at`, dates, provenance, methodology, sectors, and validation summary in research context. Disallow decision and causal fields there. Permit documented extension fields in user context while requiring `user_context_version`, `research_run_id`, preferences, notes, evidence inputs, and request responses. Require decisions, evidence ledger, risks, invalidation conditions, and source JSON paths in analysis results.

- [ ] **Step 4: Implement atomic pretty JSON writes**

Serialize with `json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"`, write to a temporary sibling file, parse it back, then replace the destination. On context regeneration, create `user_context.json` only if absent; otherwise validate and leave its bytes unchanged.

- [ ] **Step 5: Validate result-to-context traceability**

For each quantitative support item, resolve `research_json_path`, compare the copied value using exact decimal text or configured numeric tolerance, verify referenced evidence IDs exist, reject more than three Long/Short decisions, and reject any field named `weight`, `allocation`, `position_size`, or `leverage`.

- [ ] **Step 6: Run tests and commit**

Run: `python -m pytest tests/test_json_contracts.py -v`
Expected: PASS.

```bash
git add result/AI_Dir/consensus_context/contracts.py result/AI_Dir/consensus_context/serialization.py result/AI_Dir/schemas result/AI_Dir/tests/test_json_contracts.py
git commit -m "feat: define readable research JSON contracts"
```

### Task 6: Build-Context CLI and Provided-Data Integration

**Files:**
- Create: `consensus_context/cli.py`
- Modify: `consensus_context/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_context_cli.py`
- Create: `docs/consensus-json-contract.md`

**Interfaces:**
- Produces: `build_context(data_root: Path, metadata_root: Path, old_date: date, new_date: date, output_dir: Path, top_n: int = 10) -> Path`.
- Produces: CLI `build-consensus-context --data-root ... --metadata-root ... --old-date ... --new-date ... --output ...`.

- [ ] **Step 1: Write the failing end-to-end test**

```python
import json
from consensus_context.cli import main

def test_cli_emits_context_without_decisions(minimal_snapshot_args, tmp_path):
    out = tmp_path / "run"
    args = [*minimal_snapshot_args, "--output", str(out)]
    assert main(args) == 0
    research = json.loads((out / "research_context.json").read_text(encoding="utf-8"))
    assert (out / "user_context.json").exists()
    assert "decisions" not in research
    assert all("long" not in str(s).lower() for s in research["sectors"])
```

Define `minimal_snapshot_args` in `tests/conftest.py`. It writes two dated company CSVs, two dated sector CSVs, and the four metadata files under `tmp_path`, using two sectors, four horizons, and two companies per sector; it returns the complete `--data-root`, `--metadata-root`, `--old-date`, and `--new-date` argument list. Keep each numeric row different across dates so revision and contributor assertions exercise real calculations.

- [ ] **Step 2: Run the test and verify failure**

Run: `python -m pytest tests/test_context_cli.py -v`
Expected: FAIL because the CLI does not exist.

- [ ] **Step 3: Implement the orchestration boundary**

The CLI catalogs and validates sources, aligns the requested pair, calculates features and contributions, constructs research context, initializes user context only when absent, and writes `build_manifest.json`. It accepts `--top-contributors` but exposes no direction thresholds, LLM prompt, or report-rendering option.

- [ ] **Step 4: Document the editable workflow**

Document which files are generated versus editable, every top-level JSON field, how to respond to a `data_request_id`, how to regenerate without losing notes, why canonical numbers must not be manually edited, and the exact command for the provided U.S. snapshots.

- [ ] **Step 5: Run against the supplied snapshots**

Run:

```bash
python -m consensus_context.cli --data-root us --metadata-root . --old-date 2026-07-31 --new-date 2026-09-08 --output contexts/2026-09-08
```

Expected: `research_context.json`, `user_context.json`, and `build_manifest.json` are created; source CSV hashes are recorded; no SQL file, direction decision, causal claim, or non-finite JSON value is present.

- [ ] **Step 6: Verify source immutability and all tests**

Hash the four input families before and after the run and assert equality. Then run:

Run: `python -m pytest -v`
Expected: all tests PASS.

- [ ] **Step 7: Commit the verified builder**

```bash
git add result/AI_Dir/consensus_context result/AI_Dir/schemas result/AI_Dir/tests result/AI_Dir/docs/consensus-json-contract.md result/AI_Dir/contexts/2026-09-08
git commit -m "feat: build LLM-ready consensus context"
```

### Task 7: Context-Builder Review Gate

**Files:**
- Review: `consensus_context/`, `schemas/`, `tests/`, `docs/consensus-json-contract.md`, and `contexts/2026-09-08/`.

**Interfaces:**
- Consumes: the complete context-builder deliverable.
- Produces: a verified contract ready for the LLM-native skill plan.

- [ ] **Step 1: Run focused and full verification**

Run: `python -m pytest tests/test_json_contracts.py tests/test_context_cli.py -v`
Expected: PASS.
Run: `python -m pytest -v`
Expected: PASS.

- [ ] **Step 2: Inspect the generated JSON manually**

Confirm Korean text is literal, indentation is two spaces, unavailable values are `null`, sectors and contributors have stable ordering, truncation is disclosed, source hashes resolve, and `user_context.json` remains unchanged after a second build.

- [ ] **Step 3: Record the verification evidence**

Append test counts, commands, snapshot dates, hashes, generated file sizes, and unresolved data gaps to `docs/consensus-json-contract.md` under `Verification record`.

- [ ] **Step 4: Commit the verification record**

```bash
git add result/AI_Dir/docs/consensus-json-contract.md
git commit -m "docs: verify consensus JSON context builder"
```

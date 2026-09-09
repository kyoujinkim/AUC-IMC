# Consensus Report Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic Python pipeline that turns point-in-time estimate snapshots and verified evidence into a 1–3 month U.S. sector Long/Short research report without portfolio weights.

**Architecture:** A self-contained `consensus_report` package will catalog and validate snapshots, align comparable observations, calculate forward revision features and contribution attribution, apply evidence and quality gates, select up to three Long and three Short sectors, and render Markdown plus machine-readable tables. Quantitative calculations remain deterministic; narrative inputs enter only through typed evidence records, and missing inputs produce explicit user requests.

**Tech Stack:** Python 3.10+, pandas 2.x, NumPy 1.26+, SciPy 1.12+, pytest 8.x, standard-library argparse/dataclasses/json/pathlib.

**Spec:** `docs/superpowers/specs/2026-09-09-consensus-investment-report-design.md`

## Global Constraints

- Decision horizon is 1–3 months; benchmark is the S&P 500.
- Refresh monthly and evaluate 21- and 63-trading-day excess returns.
- Select at most three Long sectors and three Short sectors; never force a fixed count.
- Output direction and confidence only—never portfolio weights, leverage, or execution instructions.
- Weight forward-quarter revisions at 45%/35%/20% for `h=1/2/3`; treat `h=0` as secondary.
- Preserve pure-estimate and blended actual/estimate views as separate datasets and signals.
- Quantitative engines are deterministic; causal claims require verifiable evidence.
- Missing or unverifiable inputs generate precise user requests and cap or block confidence as specified.
- Initial thresholds are transparent priors and must not be described as statistically optimal before walk-forward validation.
- Keep all new implementation files under `result/AI_Dir`; do not refactor unrelated root-level research scripts.

## Planned File Structure

```text
result/AI_Dir/
├── pyproject.toml                         # Isolated package/test configuration
├── consensus_report/
│   ├── __init__.py                        # Public API exports
│   ├── config.py                          # Signal thresholds and weights
│   ├── models.py                          # Typed result/evidence/request records
│   ├── errors.py                          # Domain exceptions
│   ├── catalog.py                         # Snapshot discovery and date parsing
│   ├── validation.py                      # Schema, key, and coverage checks
│   ├── alignment.py                       # Point-in-time comparable panels
│   ├── features.py                        # Revision, breadth, slope, growth, normalization
│   ├── contribution.py                    # Company/industry attribution and reconciliation
│   ├── evidence.py                        # Evidence gates and missing-data requests
│   ├── selector.py                        # Long/Short/Neutral/Watch selection
│   ├── renderer.py                        # Markdown and CSV/JSON outputs
│   ├── backtest.py                        # Walk-forward evaluation
│   └── cli.py                             # End-to-end command-line entry point
└── tests/
    ├── conftest.py                        # Reusable synthetic snapshots
    ├── test_catalog_validation.py
    ├── test_alignment.py
    ├── test_features.py
    ├── test_contribution.py
    ├── test_evidence_selector.py
    ├── test_renderer_cli.py
    └── test_backtest.py
```

---

### Task 1: Package Skeleton and Domain Types

**Files:**
- Create: `pyproject.toml`
- Create: `consensus_report/__init__.py`
- Create: `consensus_report/config.py`
- Create: `consensus_report/models.py`
- Create: `consensus_report/errors.py`
- Create: `tests/conftest.py`
- Create: `tests/test_domain_types.py`

**Interfaces:**
- Produces: `SignalConfig`, `EvidenceRecord`, `DataRequest`, `SectorDecision`, `ValidationIssue`, and domain exceptions used by all later tasks.
- Produces: pytest fixtures `company_snapshots`, `sector_snapshots`, and `evidence_records`.

- [ ] **Step 1: Write the package configuration**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "consensus-report"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
  "numpy>=1.26,<3",
  "pandas>=2.0,<3",
  "scipy>=1.12,<2",
]

[project.optional-dependencies]
dev = ["pytest>=8,<9"]

[project.scripts]
consensus-report = "consensus_report.cli:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
```

- [ ] **Step 2: Write failing type/default tests**

```python
from consensus_report.config import SignalConfig
from consensus_report.models import DataRequest, EvidenceRecord


def test_signal_config_matches_approved_defaults():
    cfg = SignalConfig()
    assert cfg.forward_weights == (0.45, 0.35, 0.20)
    assert cfg.max_long == cfg.max_short == 3
    assert cfg.breadth_tolerance == 0.001


def test_data_request_records_consequence():
    req = DataRequest(
        item="sector prices",
        entities=("Information Technology",),
        date_range="2026-07-31/2026-09-08",
        reason="test price reflection",
        accepted_formats=("csv", "xlsx"),
        blocking=False,
        consequence="confidence capped at Medium",
    )
    assert req.blocking is False
```

- [ ] **Step 3: Run the tests and verify failure**

Run: `python -m pytest tests/test_domain_types.py -v`  
Expected: FAIL because `consensus_report.config` and `consensus_report.models` do not exist.

- [ ] **Step 4: Implement immutable domain records and defaults**

```python
# consensus_report/config.py
from dataclasses import dataclass


@dataclass(frozen=True)
class SignalConfig:
    forward_weights: tuple[float, float, float] = (0.45, 0.35, 0.20)
    breadth_tolerance: float = 0.001
    minimum_companies: int = 10
    minimum_coverage: float = 0.50
    concentration_limit: float = 0.70
    over_flag_limit: float = 0.10
    long_percentile: float = 0.60
    short_percentile: float = 0.40
    max_long: int = 3
    max_short: int = 3
    one_way_cost_bps: float = 10.0
```

```python
# consensus_report/models.py
from dataclasses import dataclass, field
from typing import Literal

Direction = Literal["Long", "Short", "Neutral", "Watch"]
Confidence = Literal["High", "Medium", "Low"]
ClaimStatus = Literal["Confirmed", "Corroborated", "Inferred", "Unknown"]


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    blocking: bool
    context: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceRecord:
    claim: str
    source: str
    publication_date: str
    entities: tuple[str, ...]
    driver: str
    direction: str
    horizon: str
    status: ClaimStatus


@dataclass(frozen=True)
class DataRequest:
    item: str
    entities: tuple[str, ...]
    date_range: str
    reason: str
    accepted_formats: tuple[str, ...]
    blocking: bool
    consequence: str


@dataclass(frozen=True)
class SectorDecision:
    sector: str
    sector_name: str
    direction: Direction
    confidence: Confidence
    score: float | None
    provisional: bool
    reasons: tuple[str, ...]
    risks: tuple[str, ...]
    invalidation_conditions: tuple[str, ...]
```

```python
# consensus_report/errors.py
class ConsensusReportError(Exception):
    """Base error for deterministic report generation failures."""


class SnapshotValidationError(ConsensusReportError):
    pass


class InsufficientSnapshotsError(ConsensusReportError):
    pass
```

- [ ] **Step 5: Add compact synthetic fixtures covering positive, concentrated, and deteriorating sectors**

Create `tests/conftest.py` with two dated company and sector DataFrames containing sectors `10`, `25`, `45`, and `60`, fiscal quarters `3Q26AS` through `2Q27AS`, and fields required by the metadata. Ensure sector `45` has broad upgrades, `25` has a large aggregate upgrade driven by one company, and `60` has broad downgrades.

- [ ] **Step 6: Run the tests**

Run: `python -m pytest tests/test_domain_types.py -v`  
Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add result/AI_Dir/pyproject.toml result/AI_Dir/consensus_report result/AI_Dir/tests/conftest.py result/AI_Dir/tests/test_domain_types.py
git commit -m "feat: scaffold consensus report domain"
```

---

### Task 2: Snapshot Catalog and Schema Validation

**Files:**
- Create: `consensus_report/catalog.py`
- Create: `consensus_report/validation.py`
- Create: `tests/test_catalog_validation.py`

**Interfaces:**
- Produces: `SnapshotPaths(as_of_date, company, sector_current, sector_history)`.
- Produces: `discover_snapshots(root: Path) -> list[SnapshotPaths]`.
- Produces: `validate_company_frame(df) -> list[ValidationIssue]` and `validate_sector_frame(df) -> list[ValidationIssue]`.

- [ ] **Step 1: Write failing catalog and validation tests**

```python
def test_catalog_uses_filename_date_not_mtime(tmp_path):
    (tmp_path / "mixed_model_Q_2026-09-08.csv").write_text("Code,FY,CQBtw\nA,3Q26AS,1\n")
    paths = discover_snapshots(tmp_path)
    assert paths[0].as_of_date.isoformat() == "2026-09-08"


def test_duplicate_company_key_is_blocking(company_snapshots):
    frame = company_snapshots[0]
    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    issues = validate_company_frame(duplicate)
    assert any(i.code == "duplicate_key" and i.blocking for i in issues)
```

- [ ] **Step 2: Run the targeted tests**

Run: `python -m pytest tests/test_catalog_validation.py -v`  
Expected: FAIL because catalog and validators are undefined.

- [ ] **Step 3: Implement filename parsing and grouped discovery**

Use a strict `YYYY-MM-DD` regex, require one company snapshot per date, attach optional sector files from the same date, sort by parsed date, and raise `InsufficientSnapshotsError` when fewer than two complete company snapshots exist.

```python
DATE_RE = re.compile(r"_(\d{4}-\d{2}-\d{2})\.csv$")


def parse_as_of_date(path: Path) -> date:
    match = DATE_RE.search(path.name)
    if not match:
        raise SnapshotValidationError(f"No as-of date in filename: {path.name}")
    return date.fromisoformat(match.group(1))
```

- [ ] **Step 4: Implement explicit required schemas and canonical-key checks**

Company required fields: `Code`, `FY`, `CQBtw`, `EPS_Est`, `EPS_G`, `Sector`, `LSector`, `shares`, `model`, `G`, `Over`, `name`.  
Sector required fields: `Sector`, `FY`, `CQBtw`, `earning_total`, `earning_G`, `count`, `gcount`, `acount`, `Sector_name`.

Return structured issues for missing columns, duplicate keys, invalid fiscal-quarter strings, missing identifiers, non-numeric measures, and non-finite values. Do not mutate the input frame.

- [ ] **Step 5: Run catalog/validation tests**

Run: `python -m pytest tests/test_catalog_validation.py -v`  
Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add result/AI_Dir/consensus_report/catalog.py result/AI_Dir/consensus_report/validation.py result/AI_Dir/tests/test_catalog_validation.py
git commit -m "feat: validate point-in-time snapshots"
```

---

### Task 3: Point-in-Time Alignment and Coverage Accounting

**Files:**
- Create: `consensus_report/alignment.py`
- Create: `tests/test_alignment.py`

**Interfaces:**
- Consumes: validated DataFrames from Task 2.
- Produces: `align_company_snapshots(old, new) -> AlignmentResult`.
- Produces: `align_sector_snapshots(old, new) -> AlignmentResult`.
- `AlignmentResult` contains `matched`, `old_only`, `new_only`, and `coverage` tables.

- [ ] **Step 1: Write failing exact-key and survivorship tests**

```python
def test_alignment_matches_only_exact_point_in_time_keys(company_snapshots):
    result = align_company_snapshots(*company_snapshots)
    assert set(result.matched.columns) >= {"EPS_Est_old", "EPS_Est_new"}
    assert result.matched[["Code", "FY", "CQBtw"]].duplicated().sum() == 0


def test_alignment_preserves_entry_and_exit_counts(company_snapshots):
    result = align_company_snapshots(*company_snapshots)
    assert len(result.old_only) + len(result.new_only) > 0
    assert {"old_count", "new_count", "matched_count", "matched_ratio"} <= set(result.coverage)
```

- [ ] **Step 2: Run the tests and verify failure**

Run: `python -m pytest tests/test_alignment.py -v`  
Expected: FAIL because `align_company_snapshots` is undefined.

- [ ] **Step 3: Implement immutable outer-join alignment**

Use keys `Code/FY/CQBtw` for companies and `Sector/FY/CQBtw` for sector aggregates. Suffix measure columns `_old` and `_new`, preserve identifier fields, and derive `old_only/new_only/matched` from the merge indicator.

- [ ] **Step 4: Add coverage, reporting progress, and model-mix tables**

Group company coverage by top-level sector and fiscal horizon. Group sector coverage by exact sector code and horizon. Include `count`, `acount/count`, `gcount/count`, model shares, and changes in model shares.

- [ ] **Step 5: Run alignment tests**

Run: `python -m pytest tests/test_alignment.py -v`  
Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add result/AI_Dir/consensus_report/alignment.py result/AI_Dir/tests/test_alignment.py
git commit -m "feat: align estimate snapshots point in time"
```

---

### Task 4: Revision, Breadth, Growth, and Trend Features

**Files:**
- Create: `consensus_report/features.py`
- Create: `tests/test_features.py`

**Interfaces:**
- Consumes: matched company and sector panels from Task 3 plus `SignalConfig`.
- Produces: `stable_revision(old, new) -> tuple[float, bool]`.
- Produces: `build_sector_features(company_panel, sector_panel, as_of_date, config) -> pd.DataFrame`.
- Output fields include raw horizons, `forward_revision`, `revision_slope`, breadth, medians, growth, robust ranks, quality flags, and `trend_score`.

- [ ] **Step 1: Write failing stability and weighting tests**

```python
def test_stable_revision_uses_symmetric_change_across_zero():
    value, unstable = stable_revision(-1.0, 1.0)
    assert value == pytest.approx(2.0)
    assert unstable is True


def test_forward_revision_uses_approved_weights():
    row = pd.Series({"revision_h1": 0.10, "revision_h2": 0.04, "revision_h3": -0.02})
    assert weighted_forward_revision(row, SignalConfig()) == pytest.approx(0.055)


def test_breadth_ignores_sub_tolerance_noise():
    revisions = pd.Series([0.002, -0.003, 0.0005])
    assert breadth(revisions, tolerance=0.001) == (pytest.approx(1/3), pytest.approx(1/3), pytest.approx(0.0))
```

- [ ] **Step 2: Run the tests and verify failure**

Run: `python -m pytest tests/test_features.py -v`  
Expected: FAIL because feature functions are undefined.

- [ ] **Step 3: Implement stable revision and horizon mapping**

Implement ordinary percentage revision only for stable same-sign denominators; otherwise use the symmetric scaled change and return `unstable=True`. Derive `h=0..3` by ordering the four fiscal quarters from the as-of date using the repository’s `Q#YYAS` convention; test year rollover explicitly.

- [ ] **Step 4: Implement breadth and robust descriptive statistics**

Compute up/down/net breadth, median, mean, 10% trimmed mean, mean–median gap, and per-horizon matched coverage. Keep both raw ratios and percentage display values.

- [ ] **Step 5: Implement slope, growth, normalization, and composite score**

Use `scipy.stats.linregress` on horizons `0..3` when all four revisions are valid. Normalize components with median/MAD after 5th/95th percentile winsorization; fall back to percentile rank if MAD is zero. Combine normalized components with `0.35/0.25/0.20/0.20` weights and retain component scores.

- [ ] **Step 6: Add tests for the approved behavioral examples**

Test that broad IT-like forward upgrades outrank a flat sector, a current-quarter spike with declining forward revisions is penalized by slope, and a large aggregate upgrade with negative median/breadth receives a divergence flag.

- [ ] **Step 7: Run tests**

Run: `python -m pytest tests/test_features.py -v`  
Expected: PASS.

- [ ] **Step 8: Commit**

```powershell
git add result/AI_Dir/consensus_report/features.py result/AI_Dir/tests/test_features.py
git commit -m "feat: calculate forward revision signals"
```

---

### Task 5: Company and Industry Contribution Attribution

**Files:**
- Create: `consensus_report/contribution.py`
- Create: `tests/test_contribution.py`

**Interfaces:**
- Consumes: matched company panel and sector aggregate changes.
- Produces: `attribute_contributions(company_panel, sector_panel) -> ContributionResult`.
- `ContributionResult` contains company, industry, sector, concentration, and reconciliation tables.

- [ ] **Step 1: Write failing common-share-base tests**

```python
def test_contribution_holds_old_shares_constant():
    row = pd.Series({"EPS_Est_old": 2.0, "EPS_Est_new": 2.5, "shares_old": 100.0, "shares_new": 200.0})
    assert company_profit_change(row) == pytest.approx(50.0)


def test_concentrated_upgrade_is_flagged(concentrated_company_panel, sector_panel):
    result = attribute_contributions(concentrated_company_panel, sector_panel)
    assert result.concentration.loc["25", "top5_gross_share"] > 0.70
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_contribution.py -v`  
Expected: FAIL because contribution functions are undefined.

- [ ] **Step 3: Implement company and industry attribution**

Calculate `(EPS_Est_new - EPS_Est_old) * shares_old`, preserve signed and absolute changes, derive gross shares from absolute changes, and aggregate by granular `Sector` and two-digit `LSector`.

- [ ] **Step 4: Implement concentration and reconciliation**

Calculate top-three/top-five gross shares and HHI. Compare bottom-up sector changes with supplied `earning_total_new - earning_total_old`; store absolute and relative residuals without forcing equality.

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_contribution.py -v`  
Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add result/AI_Dir/consensus_report/contribution.py result/AI_Dir/tests/test_contribution.py
git commit -m "feat: attribute sector revisions to contributors"
```

---

### Task 6: Evidence Gate and Missing-Data Requests

**Files:**
- Create: `consensus_report/evidence.py`
- Create: `tests/test_evidence_selector.py`

**Interfaces:**
- Consumes: `EvidenceRecord` objects, selected contributors, snapshot dates, and required price/evidence inputs.
- Produces: `evaluate_evidence(records, sector, old_date, new_date) -> EvidenceAssessment`.
- Produces: `build_data_requests(context) -> tuple[DataRequest, ...]`.

- [ ] **Step 1: Write failing evidence-policy tests**

```python
def test_unknown_driver_generates_specific_request():
    assessment = evaluate_evidence([], "45", date(2026, 7, 31), date(2026, 9, 8))
    assert assessment.status == "Unknown"
    assert assessment.requests[0].date_range == "2026-07-31/2026-09-08"
    assert "earnings" in assessment.requests[0].item.lower()


def test_inferred_claim_cannot_be_rendered_as_causal(evidence_records):
    inferred = replace(evidence_records[0], status="Inferred")
    assessment = evaluate_evidence([inferred], "45", date(2026, 7, 31), date(2026, 9, 8))
    assert assessment.allow_causal_language is False
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_evidence_selector.py -k evidence -v`  
Expected: FAIL because evidence evaluation is undefined.

- [ ] **Step 3: Implement temporal/source/status validation**

Accept only records with source, publication date, entities, driver, direction, horizon, and claim. Prefer records within the snapshot window; allow older structural evidence only when explicitly tagged `continuing_relevance=True`. Never upgrade `Inferred` or `Unknown` to causal language.

- [ ] **Step 4: Implement consolidated requests**

Group requests by source type and date range. Include exact entities, fields/documents, reason, acceptable formats, blocking flag, consequence, and residual conclusion. Deduplicate equivalent requests.

- [ ] **Step 5: Run evidence tests**

Run: `python -m pytest tests/test_evidence_selector.py -k evidence -v`  
Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add result/AI_Dir/consensus_report/evidence.py result/AI_Dir/tests/test_evidence_selector.py
git commit -m "feat: enforce evidence and data request policy"
```

---

### Task 7: Long/Short Direction Selector

**Files:**
- Create: `consensus_report/selector.py`
- Modify: `tests/test_evidence_selector.py`

**Interfaces:**
- Consumes: sector feature table, contribution quality, evidence assessments, price overlay availability, and `SignalConfig`.
- Produces: `select_directions(...) -> list[SectorDecision]` for all 11 sectors.

- [ ] **Step 1: Write failing direction tests**

```python
def test_selector_never_exceeds_three_per_side(selection_inputs):
    decisions = select_directions(**selection_inputs)
    assert sum(d.direction == "Long" for d in decisions) <= 3
    assert sum(d.direction == "Short" for d in decisions) <= 3


def test_selector_does_not_force_weak_signals(weak_selection_inputs):
    decisions = select_directions(**weak_selection_inputs)
    assert all(d.direction in {"Neutral", "Watch"} for d in decisions)


def test_missing_price_data_caps_confidence(selection_inputs):
    decisions = select_directions(**selection_inputs, price_overlay=None)
    assert all(d.provisional for d in decisions if d.direction in {"Long", "Short"})
    assert all(d.confidence != "High" for d in decisions)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_evidence_selector.py -k selector -v`  
Expected: FAIL because selector is undefined.

- [ ] **Step 3: Implement hard quality gates and Watch logic**

Apply minimum matched-company/coverage rules, aggregate-versus-median divergence, denominator instability, top-five concentration, duplicate/schema failures, model-mix changes, and `Over` flag confidence downgrade.

- [ ] **Step 4: Implement ranking and bounded selection**

Long requires score above the 60th percentile, positive forward revision, confirming breadth/median, and no hard failure. Short applies symmetric 40th-percentile and deterioration rules. Rank only eligible sectors, take at most three per side, and leave all other sectors Neutral or Watch.

- [ ] **Step 5: Implement confidence/provisional rules**

High requires Confirmed/Corroborated drivers plus price/valuation overlay. Missing price data caps at Medium; Unknown driver caps at Low and adds a request. Preserve exact reasons, risks, and invalidation conditions on every decision.

- [ ] **Step 6: Run selector tests**

Run: `python -m pytest tests/test_evidence_selector.py -v`  
Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add result/AI_Dir/consensus_report/selector.py result/AI_Dir/tests/test_evidence_selector.py
git commit -m "feat: select bounded sector directions"
```

---

### Task 8: Report Renderer and Command-Line Workflow

**Files:**
- Create: `consensus_report/renderer.py`
- Create: `consensus_report/cli.py`
- Modify: `consensus_report/__init__.py`
- Create: `tests/test_renderer_cli.py`

**Interfaces:**
- Consumes: decisions, feature/contribution tables, evidence ledger, validation issues, and data requests.
- Produces: `render_markdown(report: ReportBundle) -> str`.
- Produces output directory containing `report.md`, `sector_features.csv`, `company_contributions.csv`, `industry_contributions.csv`, `evidence.json`, `data_requests.json`, and `manifest.json`.

- [ ] **Step 1: Write failing renderer tests**

```python
def test_report_contains_required_sections(report_bundle):
    text = render_markdown(report_bundle)
    for heading in [
        "Executive decision", "Market earnings regime", "Sector ranking dashboard",
        "Long theses", "Short theses", "Contribution analysis",
        "Evidence ledger", "Missing-data requests", "Methodology and data quality",
    ]:
        assert f"## {heading}" in text


def test_report_bundle_has_no_weight_output(report_bundle):
    assert all(not hasattr(decision, "weight") for decision in report_bundle.decisions)
    assert report_bundle.manifest["outputs_position_sizes"] is False
```

- [ ] **Step 2: Run renderer tests and verify failure**

Run: `python -m pytest tests/test_renderer_cli.py -v`  
Expected: FAIL because renderer and CLI are undefined.

- [ ] **Step 3: Implement deterministic Markdown rendering**

Render the exact nine-section structure from the spec. Each selected sector follows `Driver → Evidence → Impact → Risk → Invalidation condition`. Cite evidence records by stable IDs; distinguish Confirmed/Corroborated/Inferred/Unknown language.

- [ ] **Step 4: Implement atomic artifact writing**

Write into a temporary sibling directory, validate all required artifacts, then rename to the final `reports/<as-of-date>/` directory. Refuse to overwrite an existing final directory unless `--replace` is explicitly passed.

- [ ] **Step 5: Implement CLI arguments**

```text
python -m consensus_report.cli \
  --data-root us \
  --old-date 2026-07-31 \
  --new-date 2026-09-08 \
  --metadata-root . \
  --evidence evidence.json \
  --prices sector_prices.csv \
  --output reports/2026-09-08
```

Make `--evidence` and `--prices` optional; omission must create provisional labels and data requests. Return exit code 2 only for blocking validation errors, not for optional evidence gaps.

- [ ] **Step 6: Add an end-to-end synthetic CLI test**

Invoke `main([...])` against temporary CSV fixtures. Assert the expected artifact set exists, no more than three Long/Short sectors appear, and missing prices produce a non-empty `data_requests.json`.

- [ ] **Step 7: Run renderer/CLI tests**

Run: `python -m pytest tests/test_renderer_cli.py -v`  
Expected: PASS.

- [ ] **Step 8: Commit**

```powershell
git add result/AI_Dir/consensus_report/renderer.py result/AI_Dir/consensus_report/cli.py result/AI_Dir/consensus_report/__init__.py result/AI_Dir/tests/test_renderer_cli.py
git commit -m "feat: render consensus investment reports"
```

---

### Task 9: Walk-Forward Backtest Module

**Files:**
- Create: `consensus_report/backtest.py`
- Create: `tests/test_backtest.py`

**Interfaces:**
- Consumes: dated decision tables and point-in-time sector/benchmark total returns.
- Produces: `run_backtest(signals, returns, config) -> BacktestResult`.
- `BacktestResult` contains observations, summary, regime, turnover, and data-quality tables.

- [ ] **Step 1: Write failing no-look-ahead and cost tests**

```python
def test_backtest_uses_returns_after_signal_date(signal_frame, return_frame):
    result = run_backtest(signal_frame, return_frame, SignalConfig())
    assert result.observations["return_start"].gt(result.observations["signal_date"]).all()


def test_transaction_cost_reduces_long_short_return(signal_frame, return_frame):
    gross = run_backtest(signal_frame, return_frame, SignalConfig(one_way_cost_bps=0))
    net = run_backtest(signal_frame, return_frame, SignalConfig(one_way_cost_bps=10))
    assert net.summary.loc["21d", "mean_long_short"] < gross.summary.loc["21d", "mean_long_short"]
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_backtest.py -v`  
Expected: FAIL because backtest is undefined.

- [ ] **Step 3: Implement 21/63-day excess-return alignment**

Require trading-date-indexed total returns. Use the first trading close after signal publication as entry, then 21- and 63-trading-day exits. Subtract S&P 500 total return for sector excess returns.

- [ ] **Step 4: Implement research-only basket and diagnostics**

Calculate equal-weight Long, equal-weight Short, and Long-minus-Short returns; turnover; costs; hit rates; rank IC; drawdown; volatility; downside capture; and Newey–West t-statistics for overlapping 63-day returns.

- [ ] **Step 5: Add explicit insufficient-history behavior**

Return a structured `DataRequest` when price history, signal history, or regime labels are insufficient. Do not emit significance claims when the effective sample is below 24 monthly observations.

- [ ] **Step 6: Run backtest tests**

Run: `python -m pytest tests/test_backtest.py -v`  
Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add result/AI_Dir/consensus_report/backtest.py result/AI_Dir/tests/test_backtest.py
git commit -m "feat: add point-in-time signal backtest"
```

---

### Task 10: Full Integration Against the Provided U.S. Snapshots

**Files:**
- Create: `tests/test_integration_us_snapshots.py`
- Create: `docs/consensus-report-usage.md`
- Modify: `pyproject.toml` only if an actually imported dependency is missing.

**Interfaces:**
- Consumes: the real metadata and U.S. CSV snapshots already present in `result/AI_Dir`.
- Produces: a reproducible provisional report under `reports/2026-09-08/` plus a documented list of missing inputs.

- [ ] **Step 1: Write a non-mutating real-data smoke test**

```python
def test_provided_us_snapshots_generate_provisional_report(tmp_path):
    exit_code = main([
        "--data-root", "us",
        "--old-date", "2026-07-31",
        "--new-date", "2026-09-08",
        "--metadata-root", ".",
        "--output", str(tmp_path / "report"),
    ])
    assert exit_code == 0
    report = (tmp_path / "report" / "report.md").read_text(encoding="utf-8")
    assert "Consensus-only provisional" in report
    assert (tmp_path / "report" / "data_requests.json").stat().st_size > 2
```

- [ ] **Step 2: Run the smoke test and fix only contract defects**

Run: `python -m pytest tests/test_integration_us_snapshots.py -v`  
Expected: PASS using the supplied data without network or paid-data access.

- [ ] **Step 3: Run the full deterministic suite**

Run: `python -m pytest -v`  
Expected: all tests PASS with zero collection errors.

- [ ] **Step 4: Generate the first provisional report**

Run:

```powershell
python -m consensus_report.cli --data-root us --old-date 2026-07-31 --new-date 2026-09-08 --metadata-root . --output reports/2026-09-08
```

Expected: report and machine-readable tables are created; price/valuation and unavailable driver documents appear as explicit data requests; no unsupported causal language or portfolio weights appear.

- [ ] **Step 5: Verify report invariants programmatically**

Run a checker that asserts: maximum three Long and three Short; all selected sectors have confidence and invalidation conditions; all causal claims resolve to evidence IDs; provisional status appears when price data is missing; canonical tables contain as-of dates.

- [ ] **Step 6: Write usage documentation**

Document required filenames, optional evidence/price schemas, the CLI example, artifact meanings, provisional behavior, and how to provide requested data. Include no credentials or provider-specific secrets.

- [ ] **Step 7: Commit**

```powershell
git add result/AI_Dir/tests/test_integration_us_snapshots.py result/AI_Dir/docs/consensus-report-usage.md result/AI_Dir/reports/2026-09-08
git commit -m "test: validate consensus report end to end"
```

---

### Task 11: Implementation Review Gate

**Files:**
- Review: all files under `consensus_report/`, `tests/`, `reports/2026-09-08/`, and `docs/consensus-report-usage.md`.

**Interfaces:**
- Consumes: completed Tasks 1–10.
- Produces: verified core engine ready for reusable-skill packaging.

- [ ] **Step 1: Confirm scope and repository hygiene**

Run: `git status --short`  
Expected: no uncommitted changes from this implementation; pre-existing unrelated user changes remain untouched.

- [ ] **Step 2: Run the full test suite once more**

Run: `python -m pytest -v`  
Expected: all tests PASS.

- [ ] **Step 3: Inspect the generated report and data requests**

Confirm the report matches the nine-section contract, presents no weights, distinguishes numerical contributors from causal drivers, and asks the user for every inaccessible input needed for a stronger conclusion.

- [ ] **Step 4: Record verification evidence**

Append exact test counts, command versions, report path, snapshot dates, and remaining data gaps to `docs/consensus-report-usage.md` under `Verification record`.

- [ ] **Step 5: Commit the verification record**

```powershell
git add result/AI_Dir/docs/consensus-report-usage.md
git commit -m "docs: record consensus report verification"
```

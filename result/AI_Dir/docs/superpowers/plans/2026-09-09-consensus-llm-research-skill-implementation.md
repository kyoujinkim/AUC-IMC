# Consensus LLM-Native Research Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a reusable `consensus-investment-report` skill that interprets the pretty JSON research context, verifies causal drivers, requests inaccessible evidence, selects up to three Long and three Short sectors, and produces traceable JSON and Markdown conclusions.

**Architecture:** The skill treats `research_context.json` as immutable calculated truth and `user_context.json` as editable, attributed input. The LLM performs exploratory comparison, primary-source research through available browser/web capabilities, competing-hypothesis analysis, and final direction judgment; deterministic helpers only build context and validate output invariants. `analysis_result.json` is the canonical decision artifact and `report.md` is its human-readable rendering.

**Tech Stack:** Codex skill format, Markdown/YAML, JSON Schema, Python 3.10+ validation helpers, browser/web research tools when available, pytest 8.x.

**Spec:** `docs/superpowers/specs/2026-09-09-consensus-investment-report-design.md`

## Global Constraints

- Start only after `2026-09-09-consensus-json-context-builder-implementation.md` passes its review gate.
- Invoke `skill-creator` for creation and validation; keep the skill automatically discoverable unless the user later requests explicit-only invocation.
- Read metadata before interpreting source fields and use the context builder for canonical calculations.
- Never edit `research_context.json`; preserve generated and user-authored provenance.
- Never mentally recompute or silently correct canonical metrics. Quote them with exact `research_json_path` references.
- Research causality from primary sources first. Search results and model recollection are not evidence.
- Mark every driver `Confirmed`, `Corroborated`, `Inferred`, or `Unknown`; causal wording is reserved for the first two.
- If evidence or inputs cannot be accessed, write one consolidated, specific request to `analysis_result.json.data_requests` and ask the user.
- Select at most three Long and three Short sectors, allow either side to be empty, and provide no portfolio weights.
- Treat current-quarter change as secondary and emphasize horizons 1–3 for a 1–3 month investment view.

## Planned File Structure

```text
result/AI_Dir/
├── skills/
│   └── consensus-investment-report/
│       ├── SKILL.md
│       ├── agents/openai.yaml
│       ├── references/
│       │   ├── json-contracts.md
│       │   ├── evidence-policy.md
│       │   └── decision-report-contract.md
│       └── scripts/
│           ├── build_context.py
│           └── verify_result.py
├── evals/
│   ├── consensus-investment-report-evals.json
│   └── consensus-investment-report-trigger-evals.json
├── tests/
│   └── test_consensus_skill.py
└── dist/
    └── consensus-investment-report.skill
```

---

### Task 1: Initialize the Skill and Define Progressive Disclosure

**Files:**
- Create: `skills/consensus-investment-report/SKILL.md`
- Create: `skills/consensus-investment-report/agents/openai.yaml`
- Create: `skills/consensus-investment-report/references/json-contracts.md`
- Create: `skills/consensus-investment-report/references/evidence-policy.md`
- Create: `skills/consensus-investment-report/references/decision-report-contract.md`
- Create: `tests/test_consensus_skill.py`

**Interfaces:**
- Consumes: `research_context.json`, `user_context.json`, and optional user-supplied documents or links.
- Produces: instructions that write `analysis_result.json` before `report.md`.

- [ ] **Step 1: Write the failing bundle test**

```python
from pathlib import Path

SKILL = Path("skills/consensus-investment-report")

def test_skill_routes_to_each_required_contract():
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    assert "name: consensus-investment-report" in text
    assert "references/json-contracts.md" in text
    assert "references/evidence-policy.md" in text
    assert "references/decision-report-contract.md" in text
    assert "research_context.json" in text
    assert "user_context.json" in text
    assert "analysis_result.json" in text
```

- [ ] **Step 2: Run the test and verify failure**

Run: `python -m pytest tests/test_consensus_skill.py -v`
Expected: FAIL because the skill files do not exist.

- [ ] **Step 3: Initialize the minimal skill structure**

Use the bundled `skill-creator` initializer with only `references` and `scripts`. Keep automatic invocation enabled. Set UI text to describe 1–3 month sector consensus research, not generic financial advice.

- [ ] **Step 4: Write the concise routing entry point**

Use this frontmatter description:

```yaml
name: consensus-investment-report
description: Analyze point-in-time company and sector earnings-consensus changes and produce an evidence-backed 1–3 month sector Long/Short research report. Use for sector revision trends, contributor analysis, causal-driver research, or updates from new estimate snapshots. Do not use for single-company valuation, portfolio sizing, or trade execution.
```

The body must route calculation requests to `scripts/build_context.py`, JSON interpretation to `json-contracts.md`, causal research to `evidence-policy.md`, and final judgment/rendering to `decision-report-contract.md`. Do not duplicate those references in the entry point.

- [ ] **Step 5: Run tests and skill validation**

Run: `python -m pytest tests/test_consensus_skill.py -v`
Expected: PASS.
Run: `python C:/Users/NHWM/.codex/skills/.system/skill-creator/scripts/quick_validate.py skills/consensus-investment-report`
Expected: validation succeeds with no scaffold placeholders.

- [ ] **Step 6: Commit the skill contract**

```bash
git add result/AI_Dir/skills/consensus-investment-report result/AI_Dir/tests/test_consensus_skill.py
git commit -m "feat: define LLM-native consensus research skill"
```

### Task 2: Implement Calculation Delegation and Result Verification

**Files:**
- Create: `skills/consensus-investment-report/scripts/build_context.py`
- Create: `skills/consensus-investment-report/scripts/verify_result.py`
- Modify: `tests/test_consensus_skill.py`

**Interfaces:**
- Produces: `build_context.py` as a narrow wrapper over `consensus_context.cli.main`.
- Produces: `verify_result.py <research-context> <analysis-result>` with exit code 0 on success and 2 on invariant failure.

- [ ] **Step 1: Add failing helper tests**

```python
import json
import subprocess
import sys

def run_verifier(tmp_path, research, analysis):
    research_path = tmp_path / "research.json"
    analysis_path = tmp_path / "analysis.json"
    research_path.write_text(json.dumps(research), encoding="utf-8")
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
    return subprocess.run(
        [sys.executable, "skills/consensus-investment-report/scripts/verify_result.py",
         str(research_path), str(analysis_path)],
        text=True, capture_output=True,
    )

def test_verifier_rejects_untraceable_number(tmp_path, valid_research, valid_analysis):
    valid_analysis["long_sectors"][0]["quantitative_support"][0]["value"] = 999
    result = run_verifier(tmp_path, valid_research, valid_analysis)
    assert result.returncode == 2
    assert "research_json_path" in result.stderr

def test_verifier_rejects_weight_fields(tmp_path, valid_research, valid_analysis):
    valid_analysis["long_sectors"][0]["weight"] = 0.2
    assert run_verifier(tmp_path, valid_research, valid_analysis).returncode == 2
```

Define `valid_research` and `valid_analysis` as concrete pytest fixtures in `tests/test_consensus_skill.py`. The research fixture contains one sector with `/sectors/0/signals/forward_revision` equal to `0.05`; the analysis fixture copies `0.05`, references that exact path, supplies one `Confirmed` evidence record, one risk, and one invalidation condition.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_consensus_skill.py -v`
Expected: FAIL because helper scripts are absent.

- [ ] **Step 3: Implement the thin builder wrapper**

Forward paths and dates unchanged to `consensus_context.cli.main`. Return its exit code and do not add calculations, direction rules, evidence text, or defaults beyond `top_n=10`.

- [ ] **Step 4: Implement the deterministic verifier**

Validate both JSON schemas; resolve every `research_json_path`; compare each copied value; ensure evidence IDs and data-request response IDs resolve; enforce distinct sector IDs, maximum-three Long/Short limits, required risk and invalidation fields, allowed evidence statuses, confidence caps, and forbidden sizing keys recursively. The verifier validates structure and provenance, not whether the investment judgment is correct.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/test_consensus_skill.py -v`
Expected: PASS.

```bash
git add result/AI_Dir/skills/consensus-investment-report/scripts result/AI_Dir/tests/test_consensus_skill.py
git commit -m "feat: validate LLM consensus decisions"
```

### Task 3: Encode the LLM-Native Research Workflow

**Files:**
- Modify: `skills/consensus-investment-report/SKILL.md`
- Modify: `skills/consensus-investment-report/references/evidence-policy.md`
- Modify: `skills/consensus-investment-report/references/decision-report-contract.md`
- Modify: `tests/test_consensus_skill.py`

**Interfaces:**
- Consumes: validated JSON contexts and accessible evidence sources.
- Produces: schema-valid `analysis_result.json` and matching `report.md`.

- [ ] **Step 1: Add failing invariant tests for the instructions**

```python
def test_skill_requires_counterevidence_and_user_requests():
    root = Path("skills/consensus-investment-report")
    content = "\n".join(p.read_text(encoding="utf-8") for p in [
        root / "SKILL.md",
        root / "references/evidence-policy.md",
        root / "references/decision-report-contract.md",
    ])
    for phrase in ["competing hypothesis", "counter-evidence", "data_requests", "research_json_path"]:
        assert phrase in content
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_consensus_skill.py -v`
Expected: FAIL until the research workflow is encoded.

- [ ] **Step 3: Specify the exploration sequence**

Require the LLM to: validate the context; inspect the market regime; compare horizon 1–3 revision direction, breadth, median, slope, growth, concentration, and flags; drill into bounded contributors; identify at least one competing explanation for each candidate; inspect price/valuation inputs when supplied; and build a provisional candidate set before external research.

- [ ] **Step 4: Specify browser and user-evidence behavior**

For each provisional candidate, search the interval between snapshot dates first and prefer filings, earnings releases, guidance, official statistics, and attributable transcripts. Record source URL/path, publication date, entity, exact supported claim, driver category, direction, and status. If a required source is inaccessible, stop that causal branch, label it `Unknown`, consolidate the exact missing documents and date range, and ask the user rather than infer access.

- [ ] **Step 5: Specify decision and rendering behavior**

Require Driver → Evidence → Impact → Risk → Invalidation for each selected sector. The LLM may override numeric-prior order only with a recorded rationale. Missing price/valuation inputs force `consensus_only_provisional=true` and confidence no higher than Medium. Render `report.md` from the completed result JSON so both artifacts agree.

- [ ] **Step 6: Run tests, validate, and commit**

Run: `python -m pytest tests/test_consensus_skill.py -v`
Expected: PASS.
Run: `python C:/Users/NHWM/.codex/skills/.system/skill-creator/scripts/quick_validate.py skills/consensus-investment-report`
Expected: validation succeeds.

```bash
git add result/AI_Dir/skills/consensus-investment-report result/AI_Dir/tests/test_consensus_skill.py
git commit -m "feat: add evidence-aware LLM research workflow"
```

### Task 4: Behavioral and Trigger Evaluation

**Files:**
- Create: `evals/consensus-investment-report-evals.json`
- Create: `evals/consensus-investment-report-trigger-evals.json`
- Create: `consensus-investment-report-workspace/iteration-1/`

**Interfaces:**
- Produces: six behavioral cases and twenty trigger cases.
- Produces: per-case artifacts and an evidence-based review summary.

- [ ] **Step 1: Define six observable behavioral cases**

Include: broad forward upgrade; concentrated upgrade with weak breadth; strong current quarter but deteriorating forward curve; broad persistent downgrade; inaccessible driver evidence; and missing price/valuation inputs. Assertions must check output structure, citations, confidence caps, missing-data requests, no weights, and no more than three sectors per side—not exact prose.

- [ ] **Step 2: Define discriminating trigger cases**

Create 10 should-trigger requests spanning new snapshots, sector trend, drivers, and monthly update, plus 10 near misses spanning single-name valuation, generic macro explanation, allocation weights, execution, and unrelated CSV formatting.

- [ ] **Step 3: Validate evaluation JSON**

Run:

```bash
python -c "import json,pathlib; a=json.loads(pathlib.Path('evals/consensus-investment-report-evals.json').read_text(encoding='utf-8')); t=json.loads(pathlib.Path('evals/consensus-investment-report-trigger-evals.json').read_text(encoding='utf-8')); assert len(a['evals'])==6; assert len(t['evals'])==20"
```

Expected: exits 0.

- [ ] **Step 4: Run independent forward tests when delegation is authorized**

Use isolated output folders and identical prompts/inputs for with-skill and baseline runs. If independent agents are not authorized or available, run the six cases inline and record that baseline comparison was not performed; never fabricate token, timing, or quality scores.

- [ ] **Step 5: Review failures and make only evidenced corrections**

Inspect actual JSON and Markdown artifacts. Fix routing or contract instructions only when a failure demonstrates ambiguity or missing guidance. Re-run the affected case and the full validation suite.

- [ ] **Step 6: Commit evaluations**

```bash
git add result/AI_Dir/evals result/AI_Dir/consensus-investment-report-workspace result/AI_Dir/skills/consensus-investment-report
git commit -m "test: evaluate LLM-native consensus research"
```

### Task 5: Package and Final Verification

**Files:**
- Create: `dist/consensus-investment-report.skill`
- Review: all skill, schema, test, evaluation, and sample output files.

**Interfaces:**
- Produces: a validated distributable skill containing only runtime resources.

- [ ] **Step 1: Run all deterministic checks**

Run: `python -m pytest -v`
Expected: all tests PASS.
Run: `python C:/Users/NHWM/.codex/skills/.system/skill-creator/scripts/quick_validate.py skills/consensus-investment-report`
Expected: validation succeeds.

- [ ] **Step 2: Execute one real provided-data research pass**

Build the 2026-07-31 to 2026-09-08 context, preserve the editable user file, perform the LLM research workflow with accessible sources, and emit `analysis_result.json` plus `report.md`. Any inaccessible evidence or missing price data must appear as a precise request, not an assumed fact.

- [ ] **Step 3: Verify the real artifacts**

Run:

```bash
python skills/consensus-investment-report/scripts/verify_result.py contexts/2026-09-08/research_context.json reports/2026-09-08/analysis_result.json
```

Expected: exit 0; report and result agree; all numeric claims resolve; no sizing fields occur.

- [ ] **Step 4: Package only runtime files**

Package `SKILL.md`, `agents/openai.yaml`, three references, and two scripts beneath one `consensus-investment-report/` root. Exclude source snapshots, contexts, reports, credentials, tests, evaluation outputs, caches, and generated evidence downloads.

- [ ] **Step 5: Report and commit final verification**

Record test count, behavioral pass rate, unavailable comparisons, package hash, and outstanding data requests. Then commit:

```bash
git add result/AI_Dir/skills/consensus-investment-report result/AI_Dir/evals result/AI_Dir/dist/consensus-investment-report.skill
git commit -m "feat: package LLM-native consensus report skill"
```

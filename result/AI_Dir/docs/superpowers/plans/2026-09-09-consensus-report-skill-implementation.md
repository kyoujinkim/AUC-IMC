# Consensus Investment Report Skill Implementation Plan

> **Superseded on 2026-09-09:** This plan assumes a deterministic direction selector and report engine. Retain it as design history only. Execute `docs/superpowers/plans/2026-09-09-consensus-llm-research-skill-implementation.md` after the JSON context-builder plan passes.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create, evaluate, refine, and package a reusable `consensus-investment-report` skill that operates the verified report engine and enforces the approved evidence and missing-data policies.

**Architecture:** The skill is a thin orchestration layer over the deterministic `consensus_report` package produced by the engine plan. `SKILL.md` routes the workflow, reference files hold the data/report/evidence contracts, and bundled scripts invoke and verify the engine; realistic with-skill and baseline evaluations test behavior before packaging.

**Tech Stack:** Codex skill format, Markdown/YAML, Python 3.10+, existing `consensus_report` package, skill-creator evaluation and review scripts.

**Spec:** `docs/superpowers/specs/2026-09-09-consensus-investment-report-design.md`

## Global Constraints

- Start this plan only after `2026-09-09-consensus-report-engine-implementation.md` passes its final review gate.
- Preserve the name `consensus-investment-report` across all iterations and packages.
- Use the deterministic engine for calculations; the skill may research and synthesize but must not recompute canonical metrics itself.
- Never fabricate inaccessible data, causal drivers, citations, prices, or valuation inputs.
- Ask the user for missing or unverifiable data using the approved consolidated request schema.
- Produce no portfolio weights; select at most three Long and three Short sectors and allow empty sides.
- Keep `SKILL.md` below 500 lines and use progressive disclosure through `references/`.
- Evaluate both qualitative output and objective assertions before packaging.
- Do not optimize the trigger description until output behavior is approved.

## Planned File Structure

```text
result/AI_Dir/
├── skills/
│   └── consensus-investment-report/
│       ├── SKILL.md
│       ├── references/
│       │   ├── data-contract.md
│       │   ├── evidence-policy.md
│       │   └── report-contract.md
│       └── scripts/
│           ├── run_report.py
│           └── verify_report.py
├── evals/
│   └── consensus-investment-report-evals.json
├── consensus-investment-report-workspace/
│   └── iteration-N/                  # Generated during evaluation
└── dist/
    └── consensus-investment-report.skill
```

---

### Task 1: Draft the Skill and Progressive-Disclosure References

**Files:**
- Create: `skills/consensus-investment-report/SKILL.md`
- Create: `skills/consensus-investment-report/references/data-contract.md`
- Create: `skills/consensus-investment-report/references/evidence-policy.md`
- Create: `skills/consensus-investment-report/references/report-contract.md`
- Create: `skills/consensus-investment-report/scripts/run_report.py`
- Create: `skills/consensus-investment-report/scripts/verify_report.py`
- Test: `tests/test_skill_bundle.py`

**Interfaces:**
- Consumes: installed/importable `consensus_report` package and user-provided metadata/snapshots/evidence/prices.
- Produces: a report artifact directory matching the core engine contract.
- Produces: `verify_report.py <report-dir>` with exit code 0 for a valid report and 2 for invariant failures.

- [ ] **Step 1: Write failing bundle tests**

```python
from pathlib import Path


SKILL = Path("skills/consensus-investment-report")


def test_skill_manifest_and_resources_exist():
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    frontmatter = text.split("---", 2)[1]
    assert "name: consensus-investment-report" in frontmatter
    assert "Long/Short" in frontmatter
    for path in [
        "references/data-contract.md", "references/evidence-policy.md",
        "references/report-contract.md", "scripts/run_report.py", "scripts/verify_report.py",
    ]:
        assert (SKILL / path).is_file()


def test_skill_md_stays_under_500_lines():
    assert len((SKILL / "SKILL.md").read_text(encoding="utf-8").splitlines()) < 500
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_skill_bundle.py -v`  
Expected: FAIL because the skill directory does not exist.

- [ ] **Step 3: Write the SKILL.md frontmatter and workflow**

Use this initial frontmatter:

```yaml
---
name: consensus-investment-report
description: Analyze point-in-time company and sector earnings-consensus snapshots and produce an evidence-backed U.S. sector Long/Short research report for a 1–3 month horizon. Use whenever the user asks which sectors to Long or Short, what changed in earnings expectations, why sector estimates improved or deteriorated, or to update a prior consensus report—even when they do not explicitly mention a report. Read metadata before interpreting columns, use the deterministic report engine, verify causal drivers, and request any inaccessible data instead of guessing.
---
```

The body must direct the agent to:

1. Discover and read all relevant metadata files.
2. Validate at least two dated snapshots.
3. Read `references/data-contract.md` before invoking the engine.
4. Run `scripts/run_report.py` rather than calculating canonical metrics in prose.
5. Inspect generated quality flags and contributor tables.
6. Research drivers from primary sources when available, following `references/evidence-policy.md`.
7. Request unavailable data in one consolidated request.
8. Rerun the engine with verified evidence/price inputs when supplied.
9. Validate outputs with `scripts/verify_report.py`.
10. Present the report according to `references/report-contract.md`.

- [ ] **Step 4: Write focused references**

Copy exact field semantics, key definitions, pure-versus-blended separation, evidence statuses, source hierarchy, missing-data request fields, report headings, confidence caps, and Long/Short limits from the approved design. Do not duplicate explanatory material across all three files.

- [ ] **Step 5: Write wrapper and verifier scripts**

`run_report.py` parses paths and delegates to `consensus_report.cli.main`. `verify_report.py` reads `manifest.json`, decision/evidence/request artifacts, and asserts maximum-three sides, no weights, evidence-linked causal claims, provisional status without prices, and invalidation conditions for selected sectors.

- [ ] **Step 6: Run bundle tests**

Run: `python -m pytest tests/test_skill_bundle.py -v`  
Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add result/AI_Dir/skills/consensus-investment-report result/AI_Dir/tests/test_skill_bundle.py
git commit -m "feat: draft consensus investment report skill"
```

---

### Task 2: Create Realistic Skill Evaluation Cases

**Files:**
- Create: `evals/consensus-investment-report-evals.json`
- Create: `consensus-investment-report-workspace/iteration-1/*/eval_metadata.json`

**Interfaces:**
- Consumes: test fixtures and the provided U.S. snapshots.
- Produces: six realistic prompts with expected outputs and objective assertions.

- [ ] **Step 1: Write six evaluation prompts**

Use these cases:

1. Broad forward upgrades with matching breadth should identify a supported Long.
2. A large aggregate upgrade concentrated in one company with negative breadth should become Watch.
3. A strong current quarter with a declining forward curve should not become an unqualified Long.
4. Persistent forward downgrades and broad negative breadth should identify a Short.
5. Missing driver documents should produce an explicit data request and prohibit causal certainty.
6. Missing price/valuation data should label directions consensus-only provisional and cap confidence.

Each prompt must name concrete snapshot paths, expected artifact directory, 1–3 month horizon, S&P 500 benchmark, and the requirement not to output weights.

- [ ] **Step 2: Add exact expected-output statements**

For every eval, describe the required direction behavior, confidence/provisional state, evidence language, data requests, and required artifacts. Avoid subjective statements such as “good report.”

- [ ] **Step 3: Add objective assertions**

Assertions must check:

- at most three Long and three Short;
- zero portfolio-weight recommendations;
- required report sections and machine-readable artifacts;
- expected Watch/Long/Short behavior for the scenario;
- evidence status attached to every driver claim;
- data request emitted for unavailable driver or price inputs;
- selected sectors include risks and invalidation conditions.

- [ ] **Step 4: Validate evaluation JSON**

Run:

```powershell
python -c "import json,pathlib; p=pathlib.Path('evals/consensus-investment-report-evals.json'); d=json.loads(p.read_text(encoding='utf-8')); assert d['skill_name']=='consensus-investment-report'; assert len(d['evals'])==6; assert all(e.get('assertions') for e in d['evals'])"
```

Expected: exit code 0.

- [ ] **Step 5: Commit**

```powershell
git add result/AI_Dir/evals/consensus-investment-report-evals.json
git commit -m "test: define consensus report skill evaluations"
```

---

### Task 3: Run With-Skill and Baseline Evaluations

**Files:**
- Create: `consensus-investment-report-workspace/iteration-1/<eval-name>/with_skill/outputs/*`
- Create: `consensus-investment-report-workspace/iteration-1/<eval-name>/without_skill/outputs/*`
- Create: per-run `timing.json`

**Interfaces:**
- Consumes: Task 2 evals and Task 1 skill.
- Produces: paired outputs for qualitative and quantitative comparison.

- [ ] **Step 1: Read the active skill-creator instructions at execution time**

Read the complete installed `skill-creator` `SKILL.md` before dispatching evaluations. Use its current schemas and scripts rather than assuming this plan’s snapshot of the workflow is exhaustive.

- [ ] **Step 2: Launch paired runs in the same execution batch**

For each eval, run one worker with `skills/consensus-investment-report` and one baseline worker without the skill. Give both the identical prompt and input files. Save only requested artifacts under their assigned output directory.

If the user selects inline execution and does not authorize subagents, execute paired cases serially while preserving isolation and identical inputs; document that this is a weaker baseline comparison.

- [ ] **Step 3: Capture timing immediately**

For each completed run, write `timing.json` with the exact `total_tokens` and `duration_ms` values from the completion notification and derive `total_duration_seconds = duration_ms / 1000`. If the execution surface does not expose either value, record the field as JSON `null` and annotate the benchmark as unavailable; never estimate it.

- [ ] **Step 4: Confirm output isolation**

Assert that each worker wrote only to its assigned `with_skill/outputs` or `without_skill/outputs` directory and that no evaluation modified the source snapshots.

- [ ] **Step 5: Commit only evaluation definitions and reusable harness changes**

Do not commit bulky generated iteration outputs unless the user explicitly wants benchmark artifacts versioned.

---

### Task 4: Grade, Aggregate, and Generate the Human Review

**Files:**
- Create: per-run `grading.json`
- Create: `consensus-investment-report-workspace/iteration-1/benchmark.json`
- Create: `consensus-investment-report-workspace/iteration-1/benchmark.md`
- Create: `consensus-investment-report-workspace/iteration-1/review.html` or launch the review server.

**Interfaces:**
- Consumes: paired outputs and assertions.
- Produces: exact assertion grades, benchmark summary, analyst observations, and a human-review surface.

- [ ] **Step 1: Grade objective assertions**

Use programmatic checks for counts, files, headings, evidence IDs, provisional labels, and missing-data requests. Each `grading.json` expectation must contain exactly `text`, `passed`, and `evidence` fields.

- [ ] **Step 2: Aggregate the benchmark**

Run from the installed skill-creator root:

```powershell
python -m scripts.aggregate_benchmark "C:\Users\NHWM\PycharmProjects\SmartConsensus\result\AI_Dir\consensus-investment-report-workspace\iteration-1" --skill-name consensus-investment-report
```

Expected: `benchmark.json` and `benchmark.md` report pass rates, duration, and token use with with-skill rows before baseline rows.

- [ ] **Step 3: Perform the analyst pass**

Identify non-discriminating assertions, flaky/high-variance cases, qualitative regressions hidden by aggregate pass rate, and time/token tradeoffs. Add concise observations to `benchmark.md`.

- [ ] **Step 4: Generate the review viewer**

Use the installed skill-creator `eval-viewer/generate_review.py`. In a headless environment generate a static file:

```powershell
python "C:\Users\NHWM\.codex\plugins\cache\claude-plugins-official\skill-creator\local\skills\skill-creator\eval-viewer\generate_review.py" "C:\Users\NHWM\PycharmProjects\SmartConsensus\result\AI_Dir\consensus-investment-report-workspace\iteration-1" --skill-name consensus-investment-report --benchmark "C:\Users\NHWM\PycharmProjects\SmartConsensus\result\AI_Dir\consensus-investment-report-workspace\iteration-1\benchmark.json" --static "C:\Users\NHWM\PycharmProjects\SmartConsensus\result\AI_Dir\consensus-investment-report-workspace\iteration-1\review.html"
```

- [ ] **Step 5: Ask the user to review outputs and benchmark**

Explain that the Outputs tab supports per-case feedback and the Benchmark tab shows formal comparisons. Wait for the user’s feedback before revising the skill.

---

### Task 5: Incorporate Feedback and Repeat Until Stable

**Files:**
- Modify: `skills/consensus-investment-report/SKILL.md` and only affected resources/scripts.
- Create: `consensus-investment-report-workspace/iteration-2/` and later iterations as needed.

**Interfaces:**
- Consumes: `feedback.json`, grading, transcripts, and benchmark observations.
- Produces: a generalized skill revision and a new complete evaluation iteration.

- [ ] **Step 1: Read user feedback and failed evidence**

Treat empty feedback as acceptable output. For complaints, inspect the corresponding transcript, output artifacts, and assertion evidence before changing instructions.

- [ ] **Step 2: Generalize the correction**

Change the smallest reusable instruction, reference, or script that addresses the underlying class of failure. Avoid prompt-specific rules and unnecessary `MUST`/`NEVER` language.

- [ ] **Step 3: Rerun all six paired evaluations**

Create a complete new iteration, retaining the same baseline for a new skill. Do not compare only previously failing cases.

- [ ] **Step 4: Generate the next review with previous-workspace comparison**

Pass `--previous-workspace` pointing to the prior iteration so the user can compare outputs and feedback.

- [ ] **Step 5: Stop only at a valid terminal condition**

Stop iterating when the user is satisfied, all feedback is empty, or changes no longer produce meaningful improvement. Record the reason in the final benchmark notes.

- [ ] **Step 6: Commit the approved skill revision**

```powershell
git add result/AI_Dir/skills/consensus-investment-report result/AI_Dir/evals/consensus-investment-report-evals.json
git commit -m "feat: refine consensus report skill"
```

---

### Task 6: Optimize Trigger Description

**Files:**
- Create: `evals/consensus-investment-report-trigger-evals.json`
- Modify: `skills/consensus-investment-report/SKILL.md`

**Interfaces:**
- Consumes: behavior-approved skill.
- Produces: 20 reviewed trigger/non-trigger queries and an improved frontmatter description selected on held-out performance.

- [ ] **Step 1: Draft 20 realistic trigger cases**

Create 10 should-trigger and 10 near-miss should-not-trigger queries. Include explicit report requests, implicit sector-revision questions, update requests, single-company EPS questions that should not trigger, generic macro commentary, and portfolio-weight requests that require a different workflow.

- [ ] **Step 2: Generate the trigger-eval review HTML**

Use the skill-creator `assets/eval_review.html` template, substitute the skill name, current description, and JSON array, then ask the user to review and export the set.

- [ ] **Step 3: Run the optimization loop if the required CLI is available**

```powershell
python -m scripts.run_loop --eval-set "C:\Users\NHWM\PycharmProjects\SmartConsensus\result\AI_Dir\evals\consensus-investment-report-trigger-evals.json" --skill-path "C:\Users\NHWM\PycharmProjects\SmartConsensus\result\AI_Dir\skills\consensus-investment-report" --model gpt-5.6-sol --max-iterations 5 --verbose
```

If the required evaluator CLI is unavailable, record that fact and skip optimization rather than fabricating scores.

- [ ] **Step 4: Apply the held-out winner**

Replace only the YAML `description`, show the user the before/after descriptions, and record train/test trigger scores.

- [ ] **Step 5: Re-run bundle tests and commit**

Run: `python -m pytest tests/test_skill_bundle.py -v`  
Expected: PASS.

```powershell
git add result/AI_Dir/skills/consensus-investment-report/SKILL.md result/AI_Dir/evals/consensus-investment-report-trigger-evals.json
git commit -m "test: optimize consensus skill triggering"
```

---

### Task 7: Package and Verify the Skill

**Files:**
- Create: `dist/consensus-investment-report.skill`
- Review: all skill files and final benchmark artifacts.

**Interfaces:**
- Consumes: approved skill directory.
- Produces: installable `.skill` package and final verification summary.

- [ ] **Step 1: Run all core and skill tests**

Run: `python -m pytest -v`  
Expected: all tests PASS.

- [ ] **Step 2: Validate no unresolved markers or local secrets**

Search skill files for unfinished-work markers, API keys, passwords, local config contents, and accidental absolute paths. Absolute paths may appear in evaluation commands but must not appear inside the packaged skill.

- [ ] **Step 3: Package with the installed skill-creator script**

```powershell
python -m scripts.package_skill "C:\Users\NHWM\PycharmProjects\SmartConsensus\result\AI_Dir\skills\consensus-investment-report"
```

Run from the installed skill-creator directory or set `PYTHONPATH` narrowly to that directory. Move the resulting package to `result/AI_Dir/dist/consensus-investment-report.skill` only after successful validation.

- [ ] **Step 4: Inspect the package contents**

Verify the archive contains `SKILL.md`, three reference files, and two scripts under one `consensus-investment-report/` root and contains no evaluation outputs, source snapshots, reports, credentials, or caches.

- [ ] **Step 5: Record final benchmark and package paths**

Report final assertion pass rate, qualitative review status, trigger score when available, package hash, and clickable local package path.

- [ ] **Step 6: Commit packaging metadata, not generated test bulk**

```powershell
git add result/AI_Dir/skills/consensus-investment-report result/AI_Dir/evals
git commit -m "feat: finalize consensus investment report skill"
```

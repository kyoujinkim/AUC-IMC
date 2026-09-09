# Consensus Investment Report — Design Specification

**Date:** 2026-09-09  
**Status:** Approved; revised for LLM-native analysis and pretty JSON exchange
**Initial market:** United States equities  
**Benchmark:** S&P 500  
**Decision horizon:** 1–3 months

## 1. Purpose

Build a repeatable research workflow that tracks changes in company and sector earnings estimates, identifies the industries and companies responsible for those changes, verifies why the changes occurred, and produces clear sector-level Long and Short candidates.

The report is a research decision aid. It does not prescribe position sizes, trade quantities, leverage, or execution instructions. A Short designation means an expectation of relative underperformance versus the S&P 500 over the next 1–3 months unless the report explicitly states otherwise.

## 2. Agreed decisions

- Use monthly report generation and monthly signal refreshes.
- Evaluate subsequent 1-month and 3-month performance relative to the S&P 500.
- Select at most three Long sectors and at most three Short sectors from the 11 top-level GICS sectors.
- Do not force the report to fill all Long or Short slots. Weak, contradictory, or insufficiently supported signals remain Neutral or Watch.
- Treat the current-quarter estimate as a secondary information-arrival signal because much of it may already be reflected in prices.
- Give primary weight to the direction, breadth, and slope of revisions across the next one to three fiscal quarters.
- Explain both what changed and why it changed. Quantitative contribution is not treated as causal explanation.
- State causal claims only when supported by directly verifiable evidence. If required information cannot be obtained or verified, request it from the user rather than inventing an explanation.
- Do not output portfolio weights.
- Preserve the supplied CSV files as immutable source data. Do not use SQL or a database as the user-facing storage layer.
- Expose the prepared research context and LLM decisions as UTF-8, two-space-indented JSON that a user can inspect and edit with an ordinary text editor.
- Keep generated facts separate from user-authored context so user edits cannot silently alter canonical calculations.

## 3. Available source data

### 3.1 Metadata

- `metadata_Q.txt`: company-level estimate fields and model descriptions.
- `metadata_CQBtw_Q.txt`: latest sector/industry estimate aggregation.
- `metadata_CQBtw_Q_sector.txt`: historical forecast-horizon sector/industry aggregation.
- `metadata_total_ts.txt`: Nelson–Siegel forecast-error curves by sector and fiscal quarter.

### 3.2 Current snapshots

- `us/mixed_model_Q_<as-of-date>.csv`: company-level estimates.
- `us/mixed_model_CQBtw_Q_<as-of-date>.csv`: current sector/industry aggregation.
- `us/mixed_model_CQBtw_Q_sector_<as-of-date>.csv`: sector/industry estimates across forecast horizons.
- `us/total_ts.csv`: forecast-error curve data.

The as-of date must be parsed from the filename and stored as a first-class field. A revision calculation requires at least two distinct snapshots. The system must never infer an as-of date from file modification time.

### 3.3 Data currently missing for a complete investment conclusion

- Point-in-time sector and benchmark total-return prices.
- Sector valuation history, preferably forward P/E and an internally consistent alternative for sectors where P/E is unstable.
- A sufficiently long archive of estimate snapshots for walk-forward testing.
- ISIN-to-ticker/security-master mapping for reliable external-source retrieval.
- Company filings, earnings releases, guidance, transcripts, and relevant macro/industry series for driver verification.

Missing data does not authorize fabricated proxies. The report must either downgrade the conclusion or issue a precise data request.

## 4. System boundaries

The initial version uses a thin deterministic data layer followed by an LLM-native research layer:

1. **Immutable source layer** — retains the supplied metadata text files and CSV snapshots without rewriting them.
2. **Context builder** — discovers snapshots, validates schemas, aligns point-in-time observations, and calculates only canonical numerical features and contribution attribution.
3. **Generated research context** — writes those facts, provenance, coverage, quality flags, and bounded contributor detail to `research_context.json`.
4. **User context** — accepts user notes, exclusions, supplementary evidence, and preferences through a separately editable `user_context.json`.
5. **LLM research agent** — reads both contexts, explores the signal dynamically, researches or accepts external evidence, tests competing explanations, and determines Long, Short, Neutral, or Watch.
6. **Decision artifacts** — writes structured judgments to `analysis_result.json` and renders the same conclusions as `report.md`.

The context builder must be deterministic and must never choose Long or Short sectors or write causal prose. The LLM must not mentally recalculate canonical metrics, alter generated facts, or state unsupported causal claims. It may compare, interpret, rank, and synthesize the supplied metrics, subject to the quality and evidence rules below.

### 4.1 Artifact flow

```text
metadata TXT + immutable source CSV
              ↓
     deterministic context builder
              ↓
 research_context.json + user_context.json
              ↓
       LLM-native research agent
              ↓
 analysis_result.json + report.md
```

No SQL database is required. Temporary in-memory data frames are an implementation detail and are not a persisted interface.

### 4.2 JSON serialization rules

All JSON artifacts use UTF-8, `ensure_ascii=false`, two-space indentation, ISO `YYYY-MM-DD` dates, stable identifiers, and deterministic key ordering within generated objects. Non-finite values are prohibited: unavailable numbers are JSON `null`, never `NaN`, `Infinity`, or string substitutes. Every generated file carries `schema_version`, `generated_at`, relevant snapshot dates, and source provenance.

`research_context.json` is generated and read-only by convention. `user_context.json` is the only intentionally user-editable input. `analysis_result.json` is regenerated by the LLM and must not be used as a source of canonical numerical facts.

## 5. Data contract and alignment

### 5.1 Canonical keys

- Company panel: `as_of_date × Code × FY × CQBtw`.
- Sector/industry panel: `as_of_date × Sector × FY × CQBtw`.
- Top-level sector is the two-digit GICS code in `LSector` or the two-digit `Sector` aggregation.
- Industry attribution uses the most granular valid GICS code present in `Sector`.

Duplicate canonical keys are a validation error. Rows with missing identifiers, fiscal quarters, or unusable estimates are retained in a rejection log and excluded from scoring.

### 5.2 Comparable snapshots

Compare observations only when entity, fiscal quarter, and forecast-horizon keys match across snapshots. Survivorship changes must be measured and reported. New or disappearing observations are not silently treated as revisions.

For each sector and horizon, report:

- old count, new count, and matched count;
- matched coverage ratio;
- announced-results ratio (`acount / count`);
- model composition and changes in model composition;
- regression-adjusted count and `Over` flag rate.

### 5.3 Earnings measures

- Use `earning_total` as the primary aggregate sector revision measure for unreported forward quarters.
- Use company `EPS_Est` for company breadth and contribution analysis.
- Use `_bld` fields only in a separately labeled blended actual/estimate view. Do not mix blended and pure-estimate measures in the same signal.
- Use the previous snapshot share count as the common share base for contribution calculations. Report material share-count changes separately rather than allowing them to masquerade as estimate revisions.

## 6. Forward revision features

Let `h = 0` denote the current fiscal quarter and `h = 1, 2, 3` the next three fiscal quarters. Let `t-1` and `t` be consecutive report snapshots.

### 6.1 Sector revision

When old and new earnings have the same sign and the old magnitude is above a stability floor:

\[
R_{s,h,t}=\frac{E_{s,h,t}}{E_{s,h,t-1}}-1
\]

When a denominator is near zero or earnings cross zero, use a symmetric scaled change:

\[
R^{scaled}_{s,h,t}=\frac{2(E_{s,h,t}-E_{s,h,t-1})}{|E_{s,h,t}|+|E_{s,h,t-1}|+\epsilon}
\]

Such observations receive a denominator-instability flag and cannot by themselves generate a high-confidence Long or Short.

### 6.2 Forward revision level

The current-quarter revision is excluded from the primary level score. The initial forward weighting is:

\[
F_{s,t}=0.45R_{s,1,t}+0.35R_{s,2,t}+0.20R_{s,3,t}
\]

These starting weights express the 1–3 month decision horizon and must later be validated out of sample rather than optimized on the currently available two snapshots.

### 6.3 Revision slope

Estimate the ordinary least-squares slope of `R` against horizon `h` across `h = 0,1,2,3`. A positive slope indicates improving revisions farther along the forward curve; a negative slope indicates deceleration. Slope is a supporting signal, not a substitute for positive forward revision level.

### 6.4 Company breadth and robustness

For each sector and forward horizon:

- Up breadth: share of matched companies with EPS revision greater than +0.1%.
- Down breadth: share with EPS revision below −0.1%.
- Net breadth: Up breadth minus Down breadth.
- Median company revision.
- Mean company revision.
- Mean–median gap and trimmed mean.

The ±0.1% tolerance prevents floating-point noise from being classified as a revision and remains configurable for later empirical validation.

### 6.5 Growth signal

Use `earning_G` as the primary sector earnings-growth measure and the change in `earning_G` across snapshots as a supporting feature. Use company `EPS_G` only for within-sector distribution and breadth analysis. Growth level and estimate revision are reported separately: high expected growth is not equivalent to improving expectations.

### 6.6 Cross-sectional normalization

Normalize features across the 11 sectors on each as-of date using winsorized percentile ranks or robust z-scores based on median and MAD. If MAD is zero, fall back to percentile rank. Preserve raw values in the report so that a normalized score never replaces the economic magnitude.

## 7. Quantitative research prior

The context builder may calculate the following transparent research prior to make cross-sector scanning easier:

\[
TrendScore_s = 0.35F_s + 0.25B_s + 0.20S_s + 0.20G_s
\]

where:

- `F`: forward revision level;
- `B`: forward company breadth and median revision;
- `S`: forward-curve slope;
- `G`: earnings-growth level and change.

Each component is cross-sectionally normalized before combination. The weights are transparent initial priors and must not be described as statistically optimal until validated on point-in-time history. This score is not an automatic trade rule: it is one input to the LLM's evidence-aware comparison, and the raw components must always accompany it.

## 8. Quantitative quality gates

A sector is ineligible for a high-confidence direction when any of the following applies:

- fewer than 10 matched companies or matched coverage below 50% for at least two of the three forward horizons;
- aggregate revision and median company revision have opposite signs for at least two forward horizons;
- a near-zero or sign-changing denominator drives the aggregate revision;
- the top five company contributions account for more than 70% of gross absolute revision and breadth does not confirm the direction;
- unresolved schema, fiscal-quarter alignment, or duplicate-key errors exist;
- a material model-composition change cannot be separated from a genuine estimate change.

An `Over` flag rate above 10% or a large regression-adjustment share is disclosed and reduces confidence by one level unless independent evidence confirms the direction.

These thresholds are conservative initial defaults and must be sensitivity-tested when historical data becomes available.

## 9. Contribution attribution

For company `i` in sector `s` and horizon `h`:

\[
\Delta Profit_{i,h}=(EPS^{t}_{i,h}-EPS^{t-1}_{i,h})\times Shares^{t-1}_i
\]

Aggregate this quantity by industry and sector. Report:

- signed contribution by industry and company;
- share of gross absolute revision;
- top-three and top-five concentration;
- contribution HHI;
- reconciliation residual versus the supplied sector `earning_total` change.

If the reconciliation residual is material, display both source totals and bottom-up totals, explain the likely data-contract difference, and do not silently force them to agree.

Attribution answers **who or what accounted for the numerical change**. It does not by itself answer **why the estimates changed**.

## 10. Driver and evidence model

### 10.1 Driver taxonomy

- Demand and revenue: volume, orders, utilization, end-market demand, backlog.
- Pricing and mix: ASP, product mix, contract repricing.
- Margin and cost: input costs, labor, logistics, productivity, operating leverage.
- Macro and financial: rates, yield curve, credit, FX, commodities, liquidity.
- Policy and event: regulation, taxation, tariffs, subsidies, litigation, M&A, supply disruption.
- Capital allocation and accounting: buybacks, share issuance, impairments, tax effects, one-time items.

### 10.2 Evidence hierarchy

Use sources in this order when available:

1. Company filings, earnings releases, and explicit management guidance.
2. Official government, regulator, exchange, and industry-body data.
3. Verifiable company transcripts and primary data-provider records.
4. High-quality secondary reporting used only as corroboration.

Search results, unattributed summaries, and model-generated text are not evidence.

### 10.3 Claim status

Every driver claim must carry one status:

- **Confirmed:** a primary source directly supports the mechanism and direction.
- **Corroborated:** at least two independent reliable sources support the same mechanism.
- **Inferred:** quantitative and contextual evidence is consistent with the driver, but direct causal confirmation is absent.
- **Unknown:** available evidence cannot support a defensible explanation.

Only Confirmed and Corroborated drivers may be phrased causally. Inferred items must use conditional language and cannot independently promote a sector to Long or Short. Unknown items generate a data request.

### 10.4 Temporal relevance

Prefer evidence published between the two estimate snapshot dates. Older evidence may be used for a structural mechanism only when its continuing relevance is explicitly established. Every cited item stores source, publication date, affected entities, driver category, direction, horizon, and the exact claim it supports.

## 11. Missing-data request protocol

When required data cannot be directly accessed or verified, stop the affected analytical branch and ask the user for a consolidated, specific input request. Each request must include:

- the exact data or document needed;
- entities, fields, and date range;
- why it is required;
- acceptable formats or links;
- whether it blocks the Long/Short conclusion or only reduces confidence;
- what the report can still conclude without it.

Example:

> The industry and company contribution to the Information Technology revision is measurable, but the reason for the change cannot be verified. Please provide earnings releases, guidance, or transcript links for the listed top contributors covering the period between the two snapshots. Without them, the sector can be labeled a quantitative Long candidate but its driver status remains Unknown and confidence is capped at Low.

Do not ask repeatedly for one field at a time when the missing requirements can be consolidated.

Requests are emitted as structured objects in `analysis_result.json.data_requests`. A user may satisfy a request by adding a file path, URL, note, or structured evidence record to `user_context.json`; the next analysis run must preserve the request ID and record whether it was resolved.

### 11.1 JSON contracts

#### `research_context.json`

This generated file contains only reproducible facts:

- run metadata, source files, hashes, snapshot dates, and schema version;
- methodology parameters and their units;
- market-level regime metrics;
- one sector object per top-level GICS sector containing raw forward revisions, breadth, medians, growth, slope, concentration, coverage, quality flags, and the research prior;
- bounded industry and company contributors for each sector, initially the top 10 positive and top 10 negative by absolute profit contribution;
- contributor totals, omitted counts, and reconciliation residuals so truncation is visible;
- rejected-row and validation summaries.

The file must not contain causal drivers, Long/Short decisions, prose copied from sources, or user overrides. If deeper contributor inspection is required, the user or LLM may request an expanded context build rather than loading every source row by default.

#### `user_context.json`

This editable file contains:

- sector exclusions or watch requirements;
- user notes and hypotheses, clearly labeled as user-provided rather than verified facts;
- supplementary evidence paths or URLs;
- optional price, valuation, or security-master inputs;
- research preferences that do not violate the fixed project constraints;
- responses keyed to outstanding data-request IDs.

Unknown fields are retained when possible so a regeneration does not erase user annotations. The context builder validates types but never silently rewrites user meaning.

#### `analysis_result.json`

This LLM-generated file contains:

- at most three Long and three Short sectors, plus Neutral and Watch sectors;
- confidence and provisional status;
- thesis, quantitative support, drivers, evidence IDs, counter-evidence, impact, risks, and invalidation conditions for every selected sector;
- an evidence ledger with `Confirmed`, `Corroborated`, `Inferred`, or `Unknown` status;
- consolidated data requests with blocking consequence and acceptable response formats;
- explicit references to the research-context run ID and user-context version used.

Canonical numbers quoted in this file must copy an exact value and JSON path from `research_context.json`. The report verifier fails values that cannot be traced to that context.

## 12. Price-reflection and valuation overlay

The consensus signal identifies where earnings expectations are changing. Price and valuation data determine whether the change is already reflected.

For each sector, the desired overlay includes:

- 1-month, 3-month, and 12-month price momentum versus the S&P 500;
- return since the prior estimate snapshot;
- forward valuation versus the sector’s own history and the market;
- optional volatility and drawdown measures.

Until these inputs are available, output `Consensus-only provisional Long/Short`, request the missing data, and cap confidence at Medium. Do not imply that an attractive earnings trend automatically produces an attractive entry point.

## 13. LLM direction-selection policy

### 13.1 Long eligibility

A sector may be selected Long by the LLM when:

- its composite score is above the 60th percentile;
- weighted forward revision is positive;
- breadth or median revision confirms the direction;
- no hard quality gate fails;
- the driver is Confirmed or Corroborated for High confidence, or Inferred/Unknown with explicit confidence limitation;
- the price/valuation overlay does not indicate extreme adverse crowding when that data is available.

### 13.2 Short eligibility

A sector may be selected Short by the LLM when:

- its composite score is below the 40th percentile;
- weighted forward revision is negative;
- down breadth or median revision confirms deterioration;
- no hard quality gate fails;
- the driver and price-reflection rules are applied symmetrically with Long selection.

### 13.3 Selection behavior

- Rank eligible sectors and select no more than three Long and three Short.
- Do not force symmetry or a fixed count.
- A sector with conflicting aggregate and breadth signals becomes Watch.
- A sector with strong numbers but missing causal or price evidence may remain a provisional direction with reduced confidence.
- The report provides direction only, not weights.
- The quantitative prior orders attention but does not mechanically determine the final direction.
- The LLM must record why a higher-ranked sector was omitted when it selects a lower-ranked sector on the same side.
- User preferences may narrow the eligible universe but may not overwrite calculated facts or convert missing evidence into verified evidence.

## 14. Report structure

The initial deliverables are `research_context.json`, `user_context.json`, `analysis_result.json`, and a Markdown report. A later implementation may render the report to HTML or PDF without changing the analytical contract.

1. **Executive decision** — Long, Short, Neutral, and Watch sectors; horizon; confidence; provisional status.
2. **Market earnings regime** — aggregate forward-growth and revision environment.
3. **Sector ranking dashboard** — raw metrics, normalized components, quality flags, and direction.
4. **Long theses** — Driver → Evidence → Impact → Risk → Invalidation condition.
5. **Short theses** — the same structure, applied symmetrically.
6. **Contribution analysis** — industries and companies responsible for each selected sector’s revision.
7. **Evidence ledger** — cited claims, dates, source class, and confidence.
8. **Missing-data requests** — consolidated blocking and non-blocking requests.
9. **Methodology and data-quality appendix** — snapshots, coverage, rejected rows, assumptions, and limitations.

Every selected sector must state what observable future event would invalidate the thesis during the 1–3 month holding period.

## 15. Error handling

- No valid pair of snapshots: do not calculate revisions; list required files.
- Schema mismatch: show missing/extra fields and the affected component.
- Duplicate keys: fail the affected panel and write a duplicate-key report.
- Insufficient coverage: retain descriptive statistics but prohibit directional selection.
- Non-finite or unstable revision: use the scaled-change rule and attach a warning.
- Missing price or driver evidence: produce a provisional quantitative view and a user data request.
- Conflicting evidence: show both sides, lower confidence, and avoid causal certainty.
- External-source failure: preserve quantitative results; mark the driver Unknown rather than replacing the source with speculation.

## 16. Backtest and validation design

Validation requires a point-in-time archive that preserves what was known on every signal date.

### 16.1 Walk-forward protocol

- Generate signals using only snapshots available on each historical as-of date.
- Observe forward 21-trading-day and 63-trading-day sector total returns.
- Subtract the contemporaneous S&P 500 total return.
- For research evaluation only, calculate an equal-weight Long basket minus equal-weight Short basket. This does not imply report-level position weights.
- Include a configurable 10-basis-point one-way transaction-cost assumption.
- Prevent revised historical data, future classifications, and post-period source evidence from entering earlier decisions.

### 16.2 Evaluation metrics

- Mean and median 1-month and 3-month excess return.
- Hit rate and Long–Short spread hit rate.
- Rank information coefficient.
- Turnover and signal persistence.
- Volatility, maximum drawdown, and downside capture.
- HAC/Newey–West t-statistics for overlapping 3-month returns.
- Results by earnings season, rate regime, growth regime, and volatility regime when sample size permits.

### 16.3 Robustness tests

- Alternative horizon weights and component weights.
- Alternative breadth tolerance and quality thresholds.
- Equal-weight versus earnings-contribution aggregation.
- Excluding sectors with high contribution concentration.
- Pure-estimate versus separately labeled blended actual/estimate views.
- Stability after transaction costs and across non-overlapping out-of-sample periods.

## 17. Testing requirements

### 17.1 Deterministic tests

- Filename date parsing and snapshot ordering.
- Required-schema and duplicate-key validation.
- Exact point-in-time joins and survivorship accounting.
- Zero, negative, and sign-changing denominator handling.
- Forward weighting, slope, breadth, median, and robust normalization.
- Industry/company contribution reconciliation.
- Quality gates and research-prior reproducibility.
- UTF-8 pretty JSON serialization, rejection of non-finite values, stable IDs, and exact provenance.
- Separation of generated facts from editable user context.

### 17.2 Narrative and evidence tests

- Every causal sentence maps to a cited evidence record.
- Inferred claims use non-causal language.
- Unknown causes generate a user request.
- Missing price data forces provisional labeling.
- Reported values reproduce canonical intermediate tables.
- Long and Short theses include risks and invalidation conditions.
- Every quoted canonical number includes a valid JSON path into the referenced `research_context.json`.
- The analysis contains no more than three Long and three Short sectors and never outputs portfolio weights.
- Re-running the LLM may change judgments but must not mutate the generated research context.

### 17.3 Acceptance scenarios

1. Broad, consistent forward upgrades produce a supported Long candidate.
2. A large aggregate upgrade driven by a few companies but negative breadth becomes Watch.
3. Current-quarter strength with a declining forward curve is not treated as an unqualified Long.
4. Persistent forward deterioration with broad downgrades produces a Short candidate.
5. Strong quantitative evidence with unavailable driver documents produces a provisional direction and an explicit user request.
6. Missing price data caps confidence and prevents an unqualified entry-timing claim.

## 18. Reusable skill scope

After this design is approved, create a reusable `consensus-investment-report` skill. It should trigger when a user asks to analyze estimate snapshots, explain sector earnings revisions, produce a 1–3 month sector Long/Short report, or update an earlier report with new snapshots.

The skill should:

- discover and read the metadata files before interpreting fields;
- validate the available snapshots and request missing inputs;
- use bundled deterministic scripts only for validation, calculation, reconciliation, and pretty JSON context generation;
- perform signal exploration, external evidence research, competing-hypothesis testing, and direction selection through the LLM-native workflow;
- read generated facts from `research_context.json` and user input from `user_context.json` without mixing their provenance;
- write decisions to `analysis_result.json` before rendering `report.md`;
- use external research only for evidence-backed driver attribution;
- generate the agreed report structure without weights;
- preserve citations, confidence labels, quality warnings, and data requests;
- be evaluated against the six acceptance scenarios above, including baseline comparisons and human review before packaging.

Skill creation and implementation planning begin only after the user reviews and approves this written specification.

## 19. Out of scope for the first implementation

- Broker integration or order execution.
- Portfolio weights, leverage, or optimization.
- Intraday signals.
- Options, futures, or single-name Long/Short recommendations.
- Fully automated acquisition of paid or credentialed datasets without an approved connection.
- Claims of statistically validated alpha before sufficient historical point-in-time data is supplied and tested.

## 20. Completion criteria

The first implementation is complete when it can:

1. Accept two valid estimate snapshots and their metadata.
2. Reproduce canonical sector revisions, forward curves, breadth, and bounded contribution detail in a valid, readable `research_context.json`.
3. Preserve user-authored material in a separate, editable `user_context.json`.
4. Use an LLM-native research pass to select no more than three evidence-qualified Long sectors and three Short sectors without outputting weights.
5. Explain quantitative contributors and distinguish them from verified causal drivers.
6. Request inaccessible or unverifiable data with precise scope and consequence.
7. Produce a schema-valid `analysis_result.json` and the agreed report with confidence, risks, invalidation conditions, and limitations.
8. Pass deterministic, provenance, and narrative acceptance tests.
9. Demonstrate walk-forward validation once the required historical price and snapshot data is available.

# Consensus Investment Report — Design Specification

**Date:** 2026-09-09  
**Status:** Proposed for user review  
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

The initial version consists of seven isolated components:

1. **Data intake and validation** — discovers snapshots, reads metadata, validates schemas, parses dates, and records missing fields.
2. **Point-in-time alignment** — matches old and new observations without look-ahead and creates comparable company, industry, and sector panels.
3. **Forward revision signal engine** — calculates revision level, breadth, growth, slope, concentration, and quality measures.
4. **Contribution engine** — attributes sector changes to industries and companies and reconciles the result with sector totals.
5. **Evidence and driver engine** — gathers or accepts verifiable evidence, maps it to drivers, distinguishes facts from inference, and creates data requests for gaps.
6. **Direction selector** — assigns Long, Short, Neutral, or Watch while enforcing confidence and quality gates.
7. **Report renderer** — produces the decision summary, evidence tables, risks, invalidation conditions, and methodology appendix.

The quantitative engines must be deterministic. A language model may synthesize narrative and classify cited evidence, but it must not calculate the canonical signal values or create unsupported causal claims.

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

## 7. Initial composite score

The starting research score is:

\[
TrendScore_s = 0.35F_s + 0.25B_s + 0.20S_s + 0.20G_s
\]

where:

- `F`: forward revision level;
- `B`: forward company breadth and median revision;
- `S`: forward-curve slope;
- `G`: earnings-growth level and change.

Each component is cross-sectionally normalized before combination. The weights are transparent initial priors. They must not be described as statistically optimal until validated on point-in-time history.

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

## 12. Price-reflection and valuation overlay

The consensus signal identifies where earnings expectations are changing. Price and valuation data determine whether the change is already reflected.

For each sector, the desired overlay includes:

- 1-month, 3-month, and 12-month price momentum versus the S&P 500;
- return since the prior estimate snapshot;
- forward valuation versus the sector’s own history and the market;
- optional volatility and drawdown measures.

Until these inputs are available, output `Consensus-only provisional Long/Short`, request the missing data, and cap confidence at Medium. Do not imply that an attractive earnings trend automatically produces an attractive entry point.

## 13. Direction-selection rules

### 13.1 Long eligibility

A sector may be selected Long when:

- its composite score is above the 60th percentile;
- weighted forward revision is positive;
- breadth or median revision confirms the direction;
- no hard quality gate fails;
- the driver is Confirmed or Corroborated for High confidence, or Inferred/Unknown with explicit confidence limitation;
- the price/valuation overlay does not indicate extreme adverse crowding when that data is available.

### 13.2 Short eligibility

A sector may be selected Short when:

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

## 14. Report structure

The initial deliverable is a Markdown report with machine-readable intermediate tables. A later implementation may render the same content to HTML or PDF without changing the analytical contract.

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
- Quality gates and maximum-three selection limits.
- No forced Long/Short behavior.

### 17.2 Narrative and evidence tests

- Every causal sentence maps to a cited evidence record.
- Inferred claims use non-causal language.
- Unknown causes generate a user request.
- Missing price data forces provisional labeling.
- Reported values reproduce canonical intermediate tables.
- Long and Short theses include risks and invalidation conditions.

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
- use bundled deterministic scripts for calculation and reconciliation;
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
2. Reproduce canonical sector revisions, forward curves, breadth, and contribution tables.
3. Select no more than three evidence-qualified Long sectors and three Short sectors without outputting weights.
4. Explain quantitative contributors and distinguish them from verified causal drivers.
5. Request inaccessible or unverifiable data with precise scope and consequence.
6. Produce the agreed report with confidence, risks, invalidation conditions, and limitations.
7. Pass deterministic and narrative acceptance tests.
8. Demonstrate walk-forward validation once the required historical price and snapshot data is available.

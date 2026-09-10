# Consensus JSON context contract

The context builder turns two dated consensus snapshots into deterministic,
LLM-ready facts. It does not select sectors, make investment decisions, write
causal explanations, or modify source CSV and metadata files.

## Build command

From `result/AI_Dir`, build the supplied U.S. snapshot pair with this exact
command:

```powershell
python -m consensus_context.cli --data-root us --metadata-root . --old-date 2026-07-31 --new-date 2026-09-08 --output contexts/2026-09-08
```

Use `--top-contributors N` to change the default top 10 positive and top 10
negative companies and industries retained per sector and horizon. `N` must be
a positive integer.

For each date the builder uses `mixed_model_Q_<date>.csv` as the company panel
and `mixed_model_CQBtw_Q_sector_<date>.csv` as the forecast-horizon sector
panel. The current aggregate `mixed_model_CQBtw_Q_<date>.csv` and undated
`total_ts.csv` may coexist in `--data-root`; because they are not inputs to
these calculations, they are left untouched and are not claimed as provenance.
The `Sector=00` all-market aggregate remains validated as source data but is
excluded from the 11-sector cross-sectional feature and rank calculation.

## Generated and editable files

| File | Owner | Regeneration behavior |
|---|---|---|
| `research_context.json` | Builder-generated, read-only facts | Replaced on every successful build |
| `build_manifest.json` | Builder-generated integrity record | Replaced on every successful build |
| `user_context.json` | User-editable input | Created only when missing; an existing valid file is never rewritten |

Canonical values in `research_context.json` must not be edited manually. They
carry calculated revisions, breadth, coverage, attribution, and source hashes;
manual changes would break reproducibility and quantitative traceability. Put
desk knowledge, preferences, and responses in `user_context.json` instead.

## `research_context.json` top-level fields

| Field | Meaning |
|---|---|
| `schema_version` | JSON contract version |
| `run_id` | Stable identifier derived from the dates and all selected source hashes |
| `generated_at` | UTC build timestamp |
| `snapshot_dates` | Requested old and new point-in-time dates |
| `provenance` | Selected CSV and required metadata paths with SHA-256 hashes |
| `methodology` | Canonical keys, forward horizons and weights, breadth tolerance, contribution formula, detail bound, and quality-gate parameters |
| `sectors` | Sector facts: revisions, forward level and slope, breadth, distribution statistics, coverage, growth, quality flags, normalized components, research prior, and bounded contributor attribution |
| `validation_summary` | Valid/rejected row counts, validation issues, and company/sector alignment coverage |

The file contains reproducible facts only. Direction labels, Long/Short/Neutral/
Watch assignments, recommendations, causal claims, risks, and portfolio sizing
belong in downstream analysis, not this document.

## `build_manifest.json` top-level fields

| Field | Meaning |
|---|---|
| `schema_version` | Manifest contract version |
| `run_id` | Same stable identifier as the research context |
| `generated_at` | Same UTC build timestamp as the research context |
| `snapshot_dates` | Requested old/new snapshot dates |
| `research_context` | Generated research-context path |
| `research_context_sha256` | SHA-256 of the generated research context |
| `user_context` | Editable user-context path |
| `user_context_sha256` | SHA-256 of the exact current user-context bytes |
| `source_sha256` | Mapping from every selected source path to its SHA-256 |

## `user_context.json` top-level fields

| Field | Meaning and editing guidance |
|---|---|
| `schema_version` | Contract version; keep at `1.0` |
| `user_context_version` | Increment whenever the user context is materially edited |
| `research_run_id` | Research build originally associated with the file; review this link after rebuilding with different inputs |
| `preferences` | Desk preferences and analysis constraints |
| `notes` | Free-form notes or structured note objects |
| `evidence_inputs` | User-supplied evidence references or structured evidence objects |
| `request_responses` | Answers keyed by a downstream `data_request_id` |
| `extensions` | Optional namespaced desk-specific data |

The schema intentionally permits documented extension fields. Keep additions
JSON-compatible and avoid non-finite numeric values such as `NaN` or infinity.
An invalid existing user context stops the build rather than being overwritten.

## Responding to a data request

When downstream `analysis_result.json` contains a `data_requests` item, copy its
exact `data_request_id` into `user_context.json.request_responses` as the key.
Record a status and the supplied answer or evidence, then increment
`user_context_version`. For example:

```json
{
  "request_responses": {
    "sector-45-guidance-2026q3": {
      "status": "answered",
      "response": "Management guidance transcript supplied in evidence_inputs.",
      "evidence_input_ids": ["issuer-transcript-2026q3"]
    }
  }
}
```

Do not delete the request identifier: it is the join key between the request
and response.

## Safe regeneration workflow

1. Edit and save `user_context.json`; increment `user_context_version`.
2. Run the build command again with the desired snapshot dates and the same
   output directory.
3. The builder recalculates `research_context.json` and `build_manifest.json`,
   validates the existing user context, and preserves its bytes unchanged.
4. Review `research_run_id` before downstream analysis if the snapshots or
   source files changed. The generated research `run_id` will change when any
   selected source content changes.

`build_manifest.json` records output paths, output hashes, snapshot dates, and
all source hashes. Use it to verify that an analysis consumed the intended
artifacts and that source inputs remained unchanged.

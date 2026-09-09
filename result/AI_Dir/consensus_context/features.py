"""Deterministic forward-consensus features and research-prior ranks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np
import pandas as pd


_FORWARD_WEIGHTS = {1: 0.45, 2: 0.35, 3: 0.20}
_COVERAGE_FIELDS = (
    "old_count",
    "new_count",
    "matched_count",
    "entrant_count",
    "exit_count",
    "matched_coverage",
)

# Conservative initial quality-gate thresholds from the research specification.
MIN_MATCHED_COVERAGE = 0.50
"""Minimum matched/new company ratio required for a forward horizon."""

MIN_MATCHED_COMPANIES = 10
"""Minimum matched-company count required for a forward horizon."""

MIN_FLAGGED_FORWARD_HORIZONS = 2
"""Number of the three forward horizons needed to trip a repeated-condition gate."""


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def stable_revision(
    old: object,
    new: object,
    *,
    stability_floor: float = 1e-9,
) -> tuple[float | None, bool]:
    """Calculate a revision, symmetrically scaling unstable denominators."""

    old_value = _finite(old)
    new_value = _finite(new)
    if old_value is None or new_value is None:
        return None, False
    unstable = abs(old_value) <= stability_floor or old_value * new_value < 0
    if not unstable:
        return new_value / old_value - 1.0, False
    denominator = abs(new_value) + abs(old_value) + stability_floor
    return 2.0 * (new_value - old_value) / denominator, True


def weighted_forward_revision(revisions: Mapping[int, float]) -> float | None:
    """Return the approved horizon 1–3 weighted level, excluding horizon 0."""

    values: dict[int, float] = {}
    for horizon in _FORWARD_WEIGHTS:
        if horizon not in revisions:
            return None
        value = _finite(revisions[horizon])
        if value is None:
            return None
        values[horizon] = value
    return sum(_FORWARD_WEIGHTS[horizon] * values[horizon] for horizon in values)


def revision_breadth(
    revisions: Sequence[float],
    tolerance: float = 0.001,
) -> dict[str, float]:
    """Return up, down, and net shares over finite matched observations."""

    values = [value for item in revisions if (value := _finite(item)) is not None]
    if not values:
        return {"up": 0.0, "down": 0.0, "net": 0.0}
    denominator = len(values)
    up = sum(value > tolerance for value in values) / denominator
    down = sum(value < -tolerance for value in values) / denominator
    return {"up": up, "down": down, "net": up - down}


def _paired_column(frame: pd.DataFrame, side: str, name: str) -> str | None:
    for candidate in (f"{side}_{name}", f"{name}_{side}"):
        if candidate in frame:
            return candidate
    return None


def _series_top_sector(frame: pd.DataFrame) -> pd.Series:
    result = pd.Series(pd.NA, index=frame.index, dtype="string")
    for name in (
        "top_sector",
        "new_LSector",
        "LSector",
        "new_Sector",
        "Sector",
        "old_LSector",
        "old_Sector",
    ):
        if name in frame:
            candidate = frame[name].astype("string").str.strip()
            result = result.fillna(candidate.where(candidate.ne("")))
    return result.str[:2]


def _horizons(frame: pd.DataFrame) -> pd.Series:
    if "CQBtw" not in frame:
        return pd.Series(pd.NA, index=frame.index, dtype="Int64")
    numeric = pd.to_numeric(frame["CQBtw"], errors="coerce")
    integral = numeric.where(numeric.mod(1).eq(0))
    return integral.astype("Int64")


def _descriptive_statistics(values: Sequence[float]) -> dict[str, float | None]:
    finite = sorted(value for item in values if (value := _finite(item)) is not None)
    if not finite:
        return {
            "mean": None,
            "median": None,
            "trimmed_mean": None,
            "mean_median_gap": None,
        }
    mean = float(np.mean(finite))
    median = float(np.median(finite))
    trim = int(len(finite) * 0.10)
    trimmed = finite[trim : len(finite) - trim] if trim else finite
    return {
        "mean": mean,
        "median": median,
        "trimmed_mean": float(np.mean(trimmed)),
        "mean_median_gap": mean - median,
    }


def _revision_slope(revisions: Mapping[int, float]) -> float | None:
    values = [_finite(revisions.get(horizon)) for horizon in range(4)]
    if any(value is None for value in values):
        return None
    x = np.arange(4, dtype=float)
    return float(np.polyfit(x, np.asarray(values, dtype=float), 1)[0])


def _mean(frame: pd.DataFrame, column: str | None) -> float | None:
    if column is None or frame.empty:
        return None
    values = pd.to_numeric(frame[column], errors="coerce")
    values = values[np.isfinite(values)]
    return float(values.mean()) if not values.empty else None


def _coverage_from_attrs(
    panel: pd.DataFrame,
    sector: str,
    horizon: int,
) -> dict[str, float] | None:
    for record in panel.attrs.get("coverage", []):
        if str(record.get("top_sector"))[:2] != sector:
            continue
        record_horizon = _finite(record.get("CQBtw"))
        if record_horizon == horizon:
            return {field: record.get(field, 0) for field in _COVERAGE_FIELDS}
    return None


def _coverage_sector_values(*panels: pd.DataFrame) -> list[str]:
    values: list[str] = []
    for panel in panels:
        for record in panel.attrs.get("coverage", []):
            value = record.get("top_sector")
            if value is None or pd.isna(value):
                continue
            text = str(value).strip()
            if text:
                values.append(text[:2])
    return values


def _coverage(
    original_panel: pd.DataFrame,
    rows: pd.DataFrame,
    sector: str,
    horizon: int,
) -> dict[str, float]:
    if not rows.empty and all(field in rows for field in _COVERAGE_FIELDS):
        result = {}
        for field in _COVERAGE_FIELDS:
            value = pd.to_numeric(rows[field], errors="coerce").dropna()
            result[field] = float(value.iloc[0]) if field == "matched_coverage" and not value.empty else (
                int(value.iloc[0]) if not value.empty else 0
            )
        return result
    from_attrs = _coverage_from_attrs(original_panel, sector, horizon)
    if from_attrs is not None:
        return {
            field: (
                float(from_attrs[field])
                if field == "matched_coverage"
                else int(from_attrs[field])
            )
            for field in _COVERAGE_FIELDS
        }
    matched = len(rows)
    return {
        "old_count": matched,
        "new_count": matched,
        "matched_count": matched,
        "entrant_count": 0,
        "exit_count": 0,
        "matched_coverage": 1.0 if matched else 0.0,
    }


def _model_composition_changed(rows: pd.DataFrame) -> bool:
    old_column = _paired_column(rows, "old", "model")
    new_column = _paired_column(rows, "new", "model")
    if old_column is None or new_column is None or rows.empty:
        return False

    groups = (
        (group for _, group in rows.groupby("_horizon", dropna=False))
        if "_horizon" in rows
        else (rows,)
    )
    for group in groups:
        old_mix = (
            group[old_column]
            .astype("string")
            .fillna("<missing>")
            .value_counts(normalize=True)
        )
        new_mix = (
            group[new_column]
            .astype("string")
            .fillna("<missing>")
            .value_counts(normalize=True)
        )
        labels = old_mix.index.union(new_mix.index)
        if any(
            not math.isclose(
                float(old_mix.get(label, 0.0)), float(new_mix.get(label, 0.0))
            )
            for label in labels
        ):
            return True
    return False


def _company_growth(rows: pd.DataFrame) -> dict[str, float | None]:
    old_column = _paired_column(rows, "old", "EPS_G")
    new_column = _paired_column(rows, "new", "EPS_G")
    old_mean = _mean(rows, old_column)
    new_mean = _mean(rows, new_column)
    changes: list[float] = []
    if old_column is not None and new_column is not None:
        for old, new in zip(rows[old_column], rows[new_column]):
            old_value, new_value = _finite(old), _finite(new)
            if old_value is not None and new_value is not None:
                changes.append(new_value - old_value)
    return {
        "old_mean": old_mean,
        "new_mean": new_mean,
        "change_mean": float(np.mean(changes)) if changes else None,
        "change_median": float(np.median(changes)) if changes else None,
    }


def _combine_available(left: float | None, right: float | None) -> float | None:
    values = [value for value in (left, right) if value is not None]
    return sum(values) / len(values) if values else None


def _robust_component_ranks(
    results: list[dict],
    component: str,
) -> tuple[dict[str, float | None], str]:
    pairs = [
        (row["sector"], _finite(row["component_values"].get(component)))
        for row in results
    ]
    valid = [(sector, value) for sector, value in pairs if value is not None]
    ranks = {sector: None for sector, _ in pairs}
    if not valid:
        return ranks, "unavailable"
    sectors, raw = zip(*valid)
    values = np.asarray(raw, dtype=float)
    if len(values) == 1:
        ranks[sectors[0]] = 0.5
        return ranks, "percentile_fallback"

    lower, upper = np.quantile(values, [0.05, 0.95])
    winsorized = np.clip(values, lower, upper)
    median = float(np.median(winsorized))
    mad = float(np.median(np.abs(winsorized - median)))
    if math.isclose(mad, 0.0, abs_tol=1e-15):
        percentile = pd.Series(winsorized).rank(method="average").to_numpy()
        percentile = (percentile - 0.5) / len(percentile)
        for sector, value in zip(sectors, percentile):
            ranks[sector] = float(value)
        return ranks, "percentile_fallback"

    robust_z = 0.67448975 * (winsorized - median) / mad
    for sector, value in zip(sectors, robust_z):
        ranks[sector] = 0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0)))
    return ranks, "median_mad"


def build_sector_features(
    company_panel: pd.DataFrame,
    sector_panel: pd.DataFrame,
) -> list[dict]:
    """Aggregate aligned panels into non-directional top-level sector facts."""

    companies = company_panel.copy()
    sectors = sector_panel.copy()
    companies["_top_sector"] = _series_top_sector(companies)
    sectors["_top_sector"] = _series_top_sector(sectors)
    companies["_horizon"] = _horizons(companies)
    sectors["_horizon"] = _horizons(sectors)

    company_sector_values = companies["_top_sector"].dropna().astype(str).tolist()
    aggregate_sector_values = sectors["_top_sector"].dropna().astype(str).tolist()
    coverage_sector_values = _coverage_sector_values(company_panel, sector_panel)
    sector_names = sorted(
        set(company_sector_values + aggregate_sector_values + coverage_sector_values)
    )
    results: list[dict] = []

    company_old_eps = _paired_column(companies, "old", "EPS_Est")
    company_new_eps = _paired_column(companies, "new", "EPS_Est")
    aggregate_old = _paired_column(sectors, "old", "earning_total")
    aggregate_new = _paired_column(sectors, "new", "earning_total")
    growth_old = _paired_column(sectors, "old", "earning_G")
    growth_new = _paired_column(sectors, "new", "earning_G")

    for sector_name in sector_names:
        company_sector = companies.loc[companies["_top_sector"].eq(sector_name)]
        aggregate_sector = sectors.loc[sectors["_top_sector"].eq(sector_name)]
        revisions: dict[int, float | None] = {}
        instability: dict[int, bool] = {}
        breadth: dict[int, dict[str, float]] = {}
        statistics: dict[int, dict[str, float | None]] = {}
        coverage: dict[int, dict[str, float]] = {}
        growth_by_horizon: dict[int, dict[str, float | None]] = {}
        company_growth: dict[int, dict[str, float | None]] = {}

        for horizon in range(4):
            company_rows = company_sector.loc[company_sector["_horizon"].eq(horizon)]
            aggregate_rows = aggregate_sector.loc[aggregate_sector["_horizon"].eq(horizon)]

            company_revisions: list[float] = []
            if company_old_eps is not None and company_new_eps is not None:
                for old_value, new_value in zip(
                    company_rows[company_old_eps], company_rows[company_new_eps]
                ):
                    value, _ = stable_revision(old_value, new_value)
                    if value is not None:
                        company_revisions.append(value)
            breadth[horizon] = revision_breadth(company_revisions)
            statistics[horizon] = _descriptive_statistics(company_revisions)
            coverage[horizon] = _coverage(
                company_panel, company_rows, sector_name, horizon
            )
            company_growth[horizon] = _company_growth(company_rows)

            old_total = None
            new_total = None
            if aggregate_old is not None and aggregate_new is not None:
                old_values = pd.to_numeric(aggregate_rows[aggregate_old], errors="coerce")
                new_values = pd.to_numeric(aggregate_rows[aggregate_new], errors="coerce")
                paired = np.isfinite(old_values) & np.isfinite(new_values)
                if paired.any():
                    old_total = float(old_values.loc[paired].sum())
                    new_total = float(new_values.loc[paired].sum())
            revisions[horizon], instability[horizon] = stable_revision(old_total, new_total)

            old_growth = _mean(aggregate_rows, growth_old)
            new_growth = _mean(aggregate_rows, growth_new)
            growth_by_horizon[horizon] = {
                "old": old_growth,
                "new": new_growth,
                "change": (
                    new_growth - old_growth
                    if old_growth is not None and new_growth is not None
                    else None
                ),
            }

        forward_revision = weighted_forward_revision(revisions)
        slope = _revision_slope(revisions)
        forward_net_breadth = weighted_forward_revision(
            {horizon: breadth[horizon]["net"] for horizon in range(1, 4)}
        )
        forward_median = weighted_forward_revision(
            {horizon: statistics[horizon]["median"] for horizon in range(1, 4)}
        )
        breadth_component = _combine_available(forward_net_breadth, forward_median)
        growth_level = weighted_forward_revision(
            {horizon: growth_by_horizon[horizon]["new"] for horizon in range(1, 4)}
        )
        growth_change = weighted_forward_revision(
            {horizon: growth_by_horizon[horizon]["change"] for horizon in range(1, 4)}
        )
        growth_component = _combine_available(growth_level, growth_change)

        low_coverage_horizons = sum(
            coverage[horizon]["matched_coverage"] < MIN_MATCHED_COVERAGE
            for horizon in range(1, 4)
        )
        low_count_horizons = sum(
            coverage[horizon]["matched_count"] < MIN_MATCHED_COMPANIES
            for horizon in range(1, 4)
        )
        conflict_horizons = sum(
            revisions[horizon] is not None
            and statistics[horizon]["median"] is not None
            and revisions[horizon] * statistics[horizon]["median"] < 0
            for horizon in range(1, 4)
        )
        quality_flags = {
            "insufficient_coverage": (
                low_coverage_horizons >= MIN_FLAGGED_FORWARD_HORIZONS
                or low_count_horizons >= MIN_FLAGGED_FORWARD_HORIZONS
            ),
            "insufficient_matched_companies": (
                low_count_horizons >= MIN_FLAGGED_FORWARD_HORIZONS
            ),
            "aggregate_median_conflict": (
                conflict_horizons >= MIN_FLAGGED_FORWARD_HORIZONS
            ),
            "denominator_instability": any(instability.values()),
            "model_composition_change": _model_composition_changed(company_sector),
        }

        results.append(
            {
                "sector": sector_name,
                "revisions": revisions,
                "forward_revision": forward_revision,
                "revision_slope": slope,
                "breadth": breadth,
                "company_revision_stats": statistics,
                "coverage": coverage,
                "growth": {
                    "by_horizon": growth_by_horizon,
                    "forward_level": growth_level,
                    "forward_change": growth_change,
                    "company": company_growth,
                },
                "denominator_instability_by_horizon": instability,
                "quality_flags": quality_flags,
                "component_values": {
                    "forward": forward_revision,
                    "breadth": breadth_component,
                    "slope": slope,
                    "growth": growth_component,
                },
            }
        )

    rank_maps: dict[str, dict[str, float | None]] = {}
    rank_methods: dict[str, str] = {}
    for component in ("forward", "breadth", "slope", "growth"):
        rank_maps[component], rank_methods[component] = _robust_component_ranks(
            results, component
        )

    for result in results:
        result["component_ranks"] = {
            component: rank_maps[component][result["sector"]]
            for component in ("forward", "breadth", "slope", "growth")
        }
        result["component_rank_methods"] = rank_methods.copy()
        ranks = result["component_ranks"]
        if all(ranks[component] is not None for component in ranks):
            result["research_prior"] = (
                0.35 * ranks["forward"]
                + 0.25 * ranks["breadth"]
                + 0.20 * ranks["slope"]
                + 0.20 * ranks["growth"]
            )
        else:
            result["research_prior"] = None
    return results

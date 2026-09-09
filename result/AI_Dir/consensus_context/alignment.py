"""Point-in-time alignment and survivorship accounting."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


_COVERAGE_FIELDS = (
    "old_count",
    "new_count",
    "matched_count",
    "entrant_count",
    "exit_count",
    "matched_coverage",
)


def _top_sector(frame: pd.DataFrame) -> pd.Series:
    """Return the two-character top-level sector without mutating *frame*."""

    values = pd.Series(pd.NA, index=frame.index, dtype="string")
    if "LSector" in frame:
        candidate = frame["LSector"].astype("string").str.strip()
        values = candidate.where(candidate.ne(""))
    if "Sector" in frame:
        candidate = frame["Sector"].astype("string").str.strip()
        values = values.fillna(candidate.where(candidate.ne("")))
    return values.str[:2]


def _counts(frame: pd.DataFrame, dimensions: list[str], name: str) -> pd.DataFrame:
    return (
        frame.groupby(dimensions, dropna=False, sort=True)
        .size()
        .rename(name)
        .reset_index()
    )


def _coverage_table(
    old: pd.DataFrame,
    new: pd.DataFrame,
    merged: pd.DataFrame,
) -> pd.DataFrame:
    dimensions = ["_coverage_sector", "FY", "CQBtw"]
    tables = [
        _counts(old, dimensions, "old_count"),
        _counts(new, dimensions, "new_count"),
    ]

    status_inputs = (
        ("both", "new__coverage_sector", "matched_count"),
        ("right_only", "new__coverage_sector", "entrant_count"),
        ("left_only", "old__coverage_sector", "exit_count"),
    )
    for status, sector_column, count_name in status_inputs:
        selected = merged.loc[merged["_merge"].eq(status), [sector_column, "FY", "CQBtw"]]
        selected = selected.rename(columns={sector_column: "_coverage_sector"})
        tables.append(_counts(selected, dimensions, count_name))

    coverage = tables[0]
    for table in tables[1:]:
        coverage = coverage.merge(table, on=dimensions, how="outer")
    count_fields = [field for field in _COVERAGE_FIELDS if field != "matched_coverage"]
    coverage[count_fields] = coverage[count_fields].fillna(0).astype(int)
    coverage["matched_coverage"] = (
        coverage["matched_count"].div(coverage["new_count"].where(coverage["new_count"].ne(0)))
    ).fillna(0.0)
    return coverage.rename(columns={"_coverage_sector": "top_sector"})


def _align(old: pd.DataFrame, new: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    missing = [key for key in keys if key not in old or key not in new]
    if missing:
        raise KeyError(f"missing canonical key columns: {', '.join(missing)}")

    old_prepared = old.copy()
    new_prepared = new.copy()
    old_prepared["_coverage_sector"] = _top_sector(old_prepared)
    new_prepared["_coverage_sector"] = _top_sector(new_prepared)

    old_names = {
        column: f"old_{column}" for column in old_prepared.columns if column not in keys
    }
    new_names = {
        column: f"new_{column}" for column in new_prepared.columns if column not in keys
    }
    merged = old_prepared.rename(columns=old_names).merge(
        new_prepared.rename(columns=new_names),
        on=list(keys),
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    coverage = _coverage_table(old_prepared, new_prepared, merged)

    panel = merged.loc[merged["_merge"].eq("both")].drop(columns="_merge").copy()
    panel["top_sector"] = panel["new__coverage_sector"].fillna(
        panel["old__coverage_sector"]
    )
    panel = panel.drop(columns=["old__coverage_sector", "new__coverage_sector"])
    panel = panel.merge(coverage, on=["top_sector", "FY", "CQBtw"], how="left")
    panel = panel.sort_values(list(keys), kind="stable").reset_index(drop=True)

    panel.attrs["coverage"] = coverage.to_dict(orient="records")
    panel.attrs["coverage_totals"] = {
        "old_count": len(old),
        "new_count": len(new),
        "matched_count": int(merged["_merge"].eq("both").sum()),
        "entrant_count": int(merged["_merge"].eq("right_only").sum()),
        "exit_count": int(merged["_merge"].eq("left_only").sum()),
        "matched_coverage": (
            float(merged["_merge"].eq("both").sum()) / len(new) if len(new) else 0.0
        ),
    }
    return panel


def align_company_snapshots(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Inner-align company rows on ``Code × FY × CQBtw``.

    Non-key columns are prefixed with ``old_`` and ``new_``. Coverage is
    repeated on matched rows and retained in ``DataFrame.attrs['coverage']``
    so zero-match sector/horizon groups remain observable.
    """

    return _align(old, new, ("Code", "FY", "CQBtw"))


def align_sector_snapshots(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Inner-align sector rows on ``Sector × FY × CQBtw``."""

    return _align(old, new, ("Sector", "FY", "CQBtw"))

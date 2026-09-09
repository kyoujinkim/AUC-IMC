"""Bounded company and industry attribution for aligned consensus panels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import pandas as pd


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _paired_column(frame: pd.DataFrame, side: str, name: str) -> str | None:
    for candidate in (f"{side}_{name}", f"{name}_{side}"):
        if candidate in frame:
            return candidate
    return None


def _text(row: Mapping[str, object], candidates: Sequence[str]) -> str | None:
    for candidate in candidates:
        value = row.get(candidate)
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _horizon(value: object) -> int | float | None:
    number = _finite(value)
    if number is None:
        return None
    return int(number) if number.is_integer() else number


def _top_sector(row: Mapping[str, object]) -> str | None:
    value = _text(
        row,
        (
            "top_sector",
            "new_LSector",
            "LSector",
            "old_LSector",
            "new_Sector",
            "Sector",
            "old_Sector",
        ),
    )
    return value[:2] if value is not None else None


def _industry(row: Mapping[str, object]) -> str | None:
    return _text(row, ("new_Sector", "Sector", "old_Sector"))


def _sum(values: Sequence[float]) -> float:
    return float(math.fsum(values))


def truncate_contributors(
    rows: Sequence[Mapping[str, object]],
    top_n: int = 10,
) -> dict:
    """Keep bounded positive and negative tails and disclose omitted totals.

    Positive rows are ordered from largest to smallest contribution; negative
    rows are ordered from most negative upward. Equal contributions use the
    canonical ``id`` converted to text as a deterministic tie-break.
    """

    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 0:
        raise ValueError("top_n must be a non-negative integer")

    prepared: list[tuple[int, dict[str, object], float, str]] = []
    for index, row in enumerate(rows):
        copied = dict(row)
        contribution = _finite(copied.get("contribution"))
        if contribution is None:
            raise ValueError("contributor contribution must be finite")
        copied["contribution"] = contribution
        prepared.append((index, copied, contribution, str(copied.get("id", ""))))

    positive = sorted(
        (item for item in prepared if item[2] > 0.0),
        key=lambda item: (-item[2], item[3]),
    )[:top_n]
    negative = sorted(
        (item for item in prepared if item[2] < 0.0),
        key=lambda item: (item[2], item[3]),
    )[:top_n]
    retained = {item[0] for item in positive + negative}
    omitted = [item for item in prepared if item[0] not in retained]

    return {
        "positive": [item[1] for item in positive],
        "negative": [item[1] for item in negative],
        "omitted_count": len(omitted),
        "omitted_signed_total": _sum([item[2] for item in omitted]),
        "all_contributors_total": _sum([item[2] for item in prepared]),
        "gross_total": _sum([abs(item[2]) for item in prepared]),
        "total_count": len(prepared),
    }


def _identity_fields(
    group: Sequence[Mapping[str, object]],
    *,
    entity_id: str,
    sector: str,
    horizon: int | float,
    industry: str | None = None,
    name_candidates: Sequence[str] = (),
) -> dict[str, object]:
    result: dict[str, object] = {
        "id": entity_id,
        "sector": sector,
        "horizon": horizon,
    }
    if industry is not None:
        result["industry"] = industry
    name = next(
        (
            value
            for row in group
            if (value := _text(row, name_candidates)) is not None
        ),
        None,
    )
    if name is not None:
        result["name"] = name
    fiscal_periods = sorted(
        {
            value
            for row in group
            if (value := _text(row, ("FY",))) is not None
        }
    )
    if len(fiscal_periods) == 1:
        result["fy"] = fiscal_periods[0]
    elif fiscal_periods:
        result["fiscal_periods"] = fiscal_periods
    return result


def _aggregate_companies(
    valid: Sequence[dict[str, Any]],
    sector: str,
    horizon: int | float,
) -> list[dict[str, object]]:
    by_company: dict[str, list[dict[str, Any]]] = {}
    for record in valid:
        by_company.setdefault(record["company_id"], []).append(record)

    result: list[dict[str, object]] = []
    for company_id in sorted(by_company):
        group = by_company[company_id]
        contribution = _sum([record["contribution"] for record in group])
        item = _identity_fields(
            [record["source"] for record in group],
            entity_id=company_id,
            sector=sector,
            horizon=horizon,
            industry=group[0]["industry"],
            name_candidates=("new_name", "name", "old_name"),
        )
        item["contribution"] = contribution
        result.append(item)
    gross_total = _sum([abs(float(item["contribution"])) for item in result])
    for item in result:
        item["absolute_contribution_share"] = (
            abs(float(item["contribution"])) / gross_total if gross_total else 0.0
        )
    return result


def _aggregate_industries(
    valid: Sequence[dict[str, Any]],
    sector: str,
    horizon: int | float,
) -> list[dict[str, object]]:
    by_industry: dict[str, list[dict[str, Any]]] = {}
    for record in valid:
        by_industry.setdefault(record["industry"], []).append(record)

    gross_total = _sum([abs(record["contribution"]) for record in valid])
    result: list[dict[str, object]] = []
    for industry_id in sorted(by_industry):
        group = by_industry[industry_id]
        contribution = _sum([record["contribution"] for record in group])
        item = _identity_fields(
            [record["source"] for record in group],
            entity_id=industry_id,
            sector=sector,
            horizon=horizon,
            name_candidates=("new_Sector_name", "Sector_name", "old_Sector_name"),
        )
        item.update(
            {
                "contribution": contribution,
                "absolute_contribution_share": (
                    abs(contribution) / gross_total if gross_total else 0.0
                ),
            }
        )
        result.append(item)
    return result


def _supplied_total_change(
    rows: Sequence[Mapping[str, object]],
    old_column: str | None,
    new_column: str | None,
) -> float | None:
    if old_column is None or new_column is None:
        return None
    for row in rows:
        old_value = _finite(row.get(old_column))
        new_value = _finite(row.get(new_column))
        if old_value is not None and new_value is not None:
            return new_value - old_value
    return None


def attribute_contributors(
    company_panel: pd.DataFrame,
    top_n: int = 10,
) -> dict[str, dict]:
    """Attribute matched EPS changes by sector, horizon, industry, and company.

    The calculation fixes the share base at the previous snapshot. Rows whose
    old/new EPS estimate or old share count is non-finite remain represented in
    their sector/horizon's ``excluded_count`` but do not enter any totals.
    """

    if company_panel.empty:
        return {}

    old_eps_column = _paired_column(company_panel, "old", "EPS_Est")
    new_eps_column = _paired_column(company_panel, "new", "EPS_Est")
    old_shares_column = _paired_column(company_panel, "old", "shares")
    missing = [
        name
        for name, column in (
            ("old EPS_Est", old_eps_column),
            ("new EPS_Est", new_eps_column),
            ("old shares", old_shares_column),
        )
        if column is None
    ]
    if missing:
        raise KeyError(f"missing aligned contribution columns: {', '.join(missing)}")

    old_total_column = _paired_column(company_panel, "old", "earning_total")
    new_total_column = _paired_column(company_panel, "new", "earning_total")
    grouped: dict[tuple[str, int | float], list[dict[str, object]]] = {}
    for source in company_panel.to_dict(orient="records"):
        sector = _top_sector(source)
        horizon = _horizon(source.get("CQBtw"))
        company_id = _text(source, ("Code",))
        industry = _industry(source)
        if sector is None or horizon is None or company_id is None or industry is None:
            continue
        grouped.setdefault((sector, horizon), []).append(source)

    result: dict[str, dict] = {}
    for (sector, horizon), sources in sorted(grouped.items()):
        valid: list[dict[str, Any]] = []
        excluded_count = 0
        for source in sources:
            old_eps = _finite(source.get(old_eps_column))
            new_eps = _finite(source.get(new_eps_column))
            old_shares = _finite(source.get(old_shares_column))
            if old_eps is None or new_eps is None or old_shares is None:
                excluded_count += 1
                continue
            valid.append(
                {
                    "company_id": _text(source, ("Code",)),
                    "industry": _industry(source),
                    "contribution": (new_eps - old_eps) * old_shares,
                    "source": source,
                }
            )

        companies = _aggregate_companies(valid, sector, horizon)
        industries = _aggregate_industries(valid, sector, horizon)
        company_detail = truncate_contributors(companies, top_n=top_n)
        industry_detail = truncate_contributors(industries, top_n=top_n)
        gross_total = company_detail["gross_total"]
        absolute_shares = [
            float(row["absolute_contribution_share"]) for row in companies
        ]
        ranked_shares = sorted(absolute_shares, reverse=True)
        supplied_change = _supplied_total_change(
            sources, old_total_column, new_total_column
        )
        bottom_up_total = company_detail["all_contributors_total"]

        identity = _identity_fields(
            sources,
            entity_id=sector,
            sector=sector,
            horizon=horizon,
            name_candidates=(
                "new_LSector_name",
                "LSector_name",
                "old_LSector_name",
            ),
        )
        horizon_result = {
            key: value for key, value in identity.items() if key != "id"
        }
        horizon_result.update(
            {
                "all_contributors_total": bottom_up_total,
                "bottom_up_total": bottom_up_total,
                "gross_total": gross_total,
                "top_three_gross_share": _sum(ranked_shares[:3]),
                "top_five_gross_share": _sum(ranked_shares[:5]),
                "contribution_hhi": _sum([share * share for share in absolute_shares]),
                "supplied_earning_total_change": supplied_change,
                "reconciliation_residual": (
                    supplied_change - bottom_up_total
                    if supplied_change is not None
                    else None
                ),
                "excluded_count": excluded_count,
                "omitted_count": company_detail["omitted_count"],
                "omitted_signed_total": company_detail["omitted_signed_total"],
                "total_count": company_detail["total_count"],
                "companies": company_detail,
                "industries": industry_detail,
            }
        )
        result.setdefault(sector, {})[horizon] = horizon_result

    return result

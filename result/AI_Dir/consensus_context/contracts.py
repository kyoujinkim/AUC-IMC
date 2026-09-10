"""Construction and validation for the three JSON artifact contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

from .serialization import write_pretty_json


SCHEMA_VERSION = "1.0"
_SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas"
_ANALYSIS_COLLECTIONS = (
    "long_sectors",
    "short_sectors",
    "neutral_sectors",
    "watch_sectors",
)
_FORBIDDEN_RESEARCH_FIELDS = {
    "cause",
    "causes",
    "decision",
    "decisions",
    "direction",
    "directions",
    "driver",
    "drivers",
    "invalidation",
    "invalidations",
    "invalidation_condition",
    "long",
    "long_sectors",
    "rating",
    "ratings",
    "recommendation",
    "recommendations",
    "risk",
    "risks",
    "short",
    "short_sectors",
    "stance",
    "stances",
    "thesis",
    "impact",
    "invalidation_conditions",
    "watch",
    "watch_sectors",
    "evidence_ledger",
    "data_requests",
    "user_override",
    "user_overrides",
}
_FORBIDDEN_SIZING_FIELDS = {"weight", "allocation", "position_size", "leverage"}


@lru_cache(maxsize=3)
def _schema(filename: str) -> dict[str, Any]:
    return json.loads((_SCHEMA_DIR / filename).read_text(encoding="utf-8"))


def _schema_errors(filename: str, value: object) -> list[str]:
    validator = Draft202012Validator(_schema(filename), format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(value), key=lambda error: list(error.path))
    formatted = []
    for error in errors:
        location = "/" + "/".join(str(part) for part in error.absolute_path)
        formatted.append(f"{location or '/'}: {error.message}")
    return formatted


def _json_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, datetime):
        instant = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        rendered = instant.isoformat(timespec="seconds")
        return rendered[:-6] + "Z" if rendered.endswith("+00:00") else rendered
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(f"non-finite Decimal is not valid JSON: {value!r}")
        if value == value.to_integral_value():
            return int(value)
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(f"Decimal is outside the finite JSON number range: {value!r}")
        return converted
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_value(value.item())
        except (TypeError, ValueError):
            pass
    return value


def _date_text(value: date | str) -> str:
    return value.isoformat() if isinstance(value, date) else str(value)


def _generated_at_text(value: datetime | str) -> str:
    return str(_json_value(value))


def _walk_fields(value: object, path: str = ""):
    if isinstance(value, Mapping):
        for key, nested in value.items():
            child_path = f"{path}/{key}"
            yield str(key), child_path
            yield from _walk_fields(nested, child_path)
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            yield from _walk_fields(nested, f"{path}/{index}")


def _nonfinite_paths(value: object, path: str = ""):
    if isinstance(value, Mapping):
        for key, nested in value.items():
            yield from _nonfinite_paths(nested, f"{path}/{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            yield from _nonfinite_paths(nested, f"{path}/{index}")
    elif isinstance(value, (float, Decimal)) and not math.isfinite(value):
        yield path or "/"


def _reject_research_judgments(value: object) -> None:
    for field, path in _walk_fields(value):
        normalized = field.casefold().replace("-", "_").replace(" ", "_")
        forbidden_prefixes = (
            "cause",
            "decision",
            "direction",
            "driver",
            "invalidation",
            "rating",
            "recommendation",
            "risk",
            "stance",
        )
        if (
            normalized in _FORBIDDEN_RESEARCH_FIELDS
            or normalized.startswith("causal")
            or any(normalized.startswith(f"{prefix}_") for prefix in forbidden_prefixes)
        ):
            raise ValueError(f"research context contains forbidden field {field!r} at {path}")


def build_research_context(
    run_id: str,
    generated_at: datetime | str,
    old_date: date | str,
    new_date: date | str,
    provenance: Sequence[Mapping[str, Any] | object],
    methodology: Mapping[str, Any],
    sectors: Sequence[Mapping[str, Any]],
    validation_summary: Mapping[str, Any],
    *,
    market: Mapping[str, Any] | None = None,
) -> dict:
    """Build a schema-valid research document containing reproducible facts only."""

    normalized_provenance = _json_value(provenance)
    normalized_sectors = _json_value(sectors)
    normalized_provenance.sort(
        key=lambda item: (
            str(item.get("kind", "")),
            str(item.get("as_of_date", "")),
            str(item.get("path", "")),
        )
    )
    normalized_sectors.sort(key=lambda item: str(item.get("sector", "")))
    result = {
        "schema_version": SCHEMA_VERSION,
        "run_id": str(run_id),
        "generated_at": _generated_at_text(generated_at),
        "snapshot_dates": {
            "old": _date_text(old_date),
            "new": _date_text(new_date),
        },
        "provenance": normalized_provenance,
        "methodology": _json_value(methodology),
        "sectors": normalized_sectors,
        "validation_summary": _json_value(validation_summary),
    }
    if market is not None:
        result["market"] = _json_value(market)

    _reject_research_judgments(result)
    nonfinite = list(_nonfinite_paths(result))
    if nonfinite:
        raise ValueError(f"research context contains non-finite value at {nonfinite[0]}")
    errors = _schema_errors("research-context.schema.json", result)
    if errors:
        raise ValueError("invalid research context: " + "; ".join(errors))
    return result


def initial_user_context(research_run_id: str) -> dict:
    """Return the minimal editable user context for a research run."""

    return {
        "schema_version": SCHEMA_VERSION,
        "user_context_version": 1,
        "research_run_id": str(research_run_id),
        "preferences": {},
        "notes": [],
        "evidence_inputs": [],
        "request_responses": {},
        "extensions": {},
    }


def ensure_user_context(path: Path, research_run_id: str) -> dict:
    """Create an initial user context only when absent; never rewrite an existing one."""

    destination = Path(path)
    if destination.exists():
        value = json.loads(destination.read_bytes().decode("utf-8"))
        nonfinite = list(_nonfinite_paths(value))
        if nonfinite:
            raise ValueError(
                f"invalid existing user context: non-finite value at {nonfinite[0]}"
            )
        errors = _schema_errors("user-context.schema.json", value)
        if errors:
            raise ValueError("invalid existing user context: " + "; ".join(errors))
        return value

    value = initial_user_context(research_run_id)
    errors = _schema_errors("user-context.schema.json", value)
    if errors:
        raise ValueError("invalid initial user context: " + "; ".join(errors))
    write_pretty_json(destination, value)
    return value


def _resolve_json_pointer(document: object, pointer: str) -> object:
    if pointer == "":
        return document
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError("must be an RFC 6901 absolute JSON pointer")
    current = document
    for raw_token in pointer[1:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            if not token.isdigit():
                raise KeyError(token)
            current = current[int(token)]
        elif isinstance(current, Mapping):
            current = current[token]
        else:
            raise KeyError(token)
    return current


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float, Decimal)) and not isinstance(value, bool)


def _numbers_match(copied: object, canonical: object, tolerance: object) -> bool:
    if not (_is_number(copied) and _is_number(canonical)):
        return False
    try:
        copied_decimal = Decimal(str(copied))
        canonical_decimal = Decimal(str(canonical))
        if not copied_decimal.is_finite() or not canonical_decimal.is_finite():
            return False
        if tolerance is None:
            return copied_decimal == canonical_decimal
        tolerance_decimal = Decimal(str(tolerance))
        return tolerance_decimal.is_finite() and tolerance_decimal >= 0 and (
            abs(copied_decimal - canonical_decimal) <= tolerance_decimal
        )
    except (InvalidOperation, ValueError):
        return False


def _decision_objects(result: Mapping[str, Any]):
    for collection in _ANALYSIS_COLLECTIONS:
        decisions = result.get(collection, [])
        if not isinstance(decisions, list):
            continue
        for index, decision in enumerate(decisions):
            if isinstance(decision, Mapping):
                yield collection, index, decision


def _evidence_references(value: object, path: str):
    if isinstance(value, Mapping):
        for field, nested in value.items():
            child_path = f"{path}/{field}"
            if field == "evidence_ids" and isinstance(nested, list):
                yield child_path, nested
            yield from _evidence_references(nested, child_path)
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            yield from _evidence_references(nested, f"{path}/{index}")


def validate_analysis_result(result: dict, research: dict) -> list[str]:
    """Return structural, safety, provenance, and traceability errors."""

    errors = _schema_errors("analysis-result.schema.json", result)
    errors.extend(
        f"{path}: non-finite values are prohibited"
        for path in _nonfinite_paths(result)
    )

    if result.get("research_run_id") != research.get("run_id"):
        errors.append(
            "/research_run_id: does not match the referenced research context run_id"
        )

    for field, path in _walk_fields(result):
        if field.casefold() in _FORBIDDEN_SIZING_FIELDS:
            errors.append(f"{path}: forbidden sizing field {field!r}")

    seen_sectors: dict[str, str] = {}
    for collection in ("long_sectors", "short_sectors"):
        decisions = result.get(collection)
        if isinstance(decisions, list) and len(decisions) > 3:
            errors.append(f"/{collection}: may contain at most 3 sectors")

    evidence_ledger = result.get("evidence_ledger", [])
    evidence_ids: set[str] = set()
    if isinstance(evidence_ledger, list):
        for index, evidence in enumerate(evidence_ledger):
            if not isinstance(evidence, Mapping):
                continue
            evidence_id = evidence.get("evidence_id")
            if isinstance(evidence_id, str):
                if evidence_id in evidence_ids:
                    errors.append(
                        f"/evidence_ledger/{index}/evidence_id: duplicate evidence ID {evidence_id!r}"
                    )
                evidence_ids.add(evidence_id)

    for collection, index, decision in _decision_objects(result):
        location = f"/{collection}/{index}"
        sector = decision.get("sector")
        if isinstance(sector, str):
            if sector in seen_sectors:
                errors.append(
                    f"{location}/sector: sector IDs must be unique; {sector!r} also appears at {seen_sectors[sector]}"
                )
            else:
                seen_sectors[sector] = location

        for required_list in ("risks", "invalidation_conditions"):
            value = decision.get(required_list)
            if not isinstance(value, list) or not value:
                errors.append(f"{location}/{required_list}: must contain at least one item")

        for path, referenced in _evidence_references(decision, location):
            for evidence_id in referenced:
                if isinstance(evidence_id, str) and evidence_id not in evidence_ids:
                    errors.append(f"{path}: unknown evidence ID {evidence_id!r}")

        supports = decision.get("quantitative_support", [])
        if not isinstance(supports, list):
            continue
        for support_index, support in enumerate(supports):
            if not isinstance(support, Mapping):
                continue
            support_location = f"{location}/quantitative_support/{support_index}"
            pointer = support.get("research_json_path")
            try:
                canonical = _resolve_json_pointer(research, pointer)
            except (IndexError, KeyError, TypeError, ValueError) as exc:
                errors.append(
                    f"{support_location}/research_json_path: cannot resolve {pointer!r}: {exc}"
                )
                continue
            copied = support.get("value")
            if not (_is_number(copied) and _is_number(canonical)):
                errors.append(
                    f"{support_location}/research_json_path: non-numeric quantitative "
                    f"support value or canonical value at {pointer!r}"
                )
                continue
            if not _numbers_match(
                copied, canonical, support.get("numeric_tolerance")
            ):
                errors.append(
                    f"{support_location}/research_json_path: copied value {copied!r} "
                    f"does not match canonical value {canonical!r} at {pointer!r}"
                )

    return list(dict.fromkeys(errors))

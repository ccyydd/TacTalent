"""Record provider-reported usage and charges; never infer prices or missing costs."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

SCHEMA = "provider_api_usage_v1"
COUNTERS = (
    "attempts",
    "requests_with_tokens",
    "requests_with_cost",
    "missing_records",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "reasoning_tokens",
    "cached_tokens",
    "cache_write_tokens",
)


def safe_usage(value):
    """Allowlist numeric billing data, excluding messages and credentials."""
    if not isinstance(value, dict):
        return {}
    result = {}

    def count(x):
        return isinstance(x, int) and not isinstance(x, bool) and x >= 0

    def money(x):
        return (
            isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x >= 0
        )

    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if count(value.get(name)):
            result[name] = value[name]
    for name in ("prompt_tokens_details", "completion_tokens_details"):
        details = value.get(name)
        if isinstance(details, dict):
            result[name] = {
                key: item
                for key, item in details.items()
                if key
                in (
                    "cached_tokens",
                    "cache_write_tokens",
                    "reasoning_tokens",
                    "audio_tokens",
                    "video_tokens",
                    "image_tokens",
                    "accepted_prediction_tokens",
                    "rejected_prediction_tokens",
                )
                and count(item)
            }
    if count(value.get("cacheReadInputTokensCompatible")):
        result["cacheReadInputTokensCompatible"] = value["cacheReadInputTokensCompatible"]
        result.setdefault("prompt_tokens_details", {}).setdefault(
            "cached_tokens", value["cacheReadInputTokensCompatible"]
        )
    if money(value.get("cost")):
        result["cost"] = value["cost"]
    currency = value.get("currency")
    if (
        isinstance(currency, str)
        and len(currency) == 3
        and currency.isascii()
        and currency.isalpha()
    ):
        result["currency"] = currency.upper()
    details = value.get("cost_details")
    if isinstance(details, dict):
        result["cost_details"] = {
            key: item
            for key, item in details.items()
            if key
            in (
                "upstream_inference_cost",
                "upstream_inference_prompt_cost",
                "upstream_inference_completions_cost",
            )
            and money(item)
        }
    if isinstance(value.get("is_byok"), bool):
        result["is_byok"] = value["is_byok"]
    return result


def response_metadata(payload):
    """Keep the generation ID for later billing reconciliation."""
    if not isinstance(payload, dict):
        return {"usage": {}}
    result = {"usage": safe_usage(payload.get("usage"))}
    for source, target in (
        ("id", "response_id"),
        ("model", "response_model"),
        ("provider", "response_provider"),
    ):
        if isinstance(payload.get(source), str):
            result[target] = payload[source]
    return result


def _cost_fields(result, costs, unclassified, currencies):
    currencies.update(costs)
    result["known_cost_by_currency"] = {unit: float(value) for unit, value in costs.items()}
    result["unclassified_cost_count"] = unclassified
    result["known_cost_usd"] = float(costs.get("USD", Decimal(0)))
    result["currency"] = next(iter(currencies)) if len(currencies) == 1 else None
    result["cost_usd"] = (
        result["known_cost_usd"]
        if (result["cost_complete"] and result["currency"] == "USD")
        else None
    )
    return result


def summarize_attempts(records):
    """Sum returned charges by currency without estimating or converting money."""
    result = {name: 0 for name in COUNTERS}
    result["attempts"] = len(records)
    costs, currencies, unclassified = {}, set(), 0
    for record in records:
        usage = safe_usage(record.get("usage"))
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            result[name] += usage.get(name, 0)
        result["requests_with_tokens"] += all(
            name in usage for name in ("prompt_tokens", "completion_tokens", "total_tokens")
        )
        result["requests_with_cost"] += "cost" in usage
        currency = usage.get("currency")
        if currency is None and record.get("api_provider") != "gateway":
            currency = "USD"  # Original OpenRouter accounting contract.
        if currency is not None:
            currencies.add(currency)
        if "cost" in usage:
            if currency is None:
                unclassified += 1
            else:
                costs[currency] = costs.get(currency, Decimal(0)) + Decimal(str(usage["cost"]))
        for name in ("cached_tokens", "cache_write_tokens"):
            result[name] += usage.get("prompt_tokens_details", {}).get(name, 0)
        result["reasoning_tokens"] += usage.get("completion_tokens_details", {}).get(
            "reasoning_tokens", 0
        )
    if not records:
        currencies.add("USD")
    result.update(
        schema_version=SCHEMA,
        source="provider_reported",
        tokens_complete=result["requests_with_tokens"] == len(records),
        cost_complete=result["requests_with_cost"] == len(records) and unclassified == 0,
    )
    return _cost_fields(result, costs, unclassified, currencies)

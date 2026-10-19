"""Gateway error-envelope compatibility; the vendored RoboICL retry policy stays intact."""

import json
from functools import wraps

MAX_ERROR_CHARS = 8192
MAX_ENVELOPE_DEPTH = 8
_FIELDS = ("type", "code", "reason", "message", "param")


def error_details(value):
    """Unwrap structured errors only; never classify an error by matching prose."""
    for _ in range(MAX_ENVELOPE_DEPTH):
        if not isinstance(value, dict):
            return {}
        nested = value.get("error")
        if isinstance(nested, dict):
            value = nested
            continue
        response = value.get("response")
        if isinstance(response, dict) and isinstance(response.get("error"), dict):
            value = response["error"]
            continue
        # An explicit outer code has priority. Generic gateway errors sometimes
        # put the actual provider JSON object in their message string instead.
        message = value.get("message")
        if not value.get("code") and isinstance(message, str) and len(message) <= MAX_ERROR_CHARS:
            try:
                nested = json.loads(message)
            except (TypeError, ValueError):
                nested = None
            if isinstance(nested, dict) and (
                isinstance(nested.get("error"), dict) or nested.get("code")
            ):
                value = nested
                continue
        return {key: value[key] for key in _FIELDS if key in value}
    return {}


def http_error_details(text):
    """Handle ordinary JSON and complete SSE error records inside HTTP failures."""
    if not isinstance(text, str) or len(text) > MAX_ERROR_CHARS:
        return None, None
    try:
        event = json.loads(text)
    except (TypeError, ValueError):
        event = None
        data = []
        # No appended blank line: an unfinished SSE event is not trusted.
        for line in text.splitlines():
            if line.startswith("data:"):
                data.append(line[5:].lstrip(" "))
            elif not line and data:
                raw, data = "\n".join(data), []
                try:
                    candidate = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                if isinstance(candidate, dict) and candidate.get("type") in (
                    "error",
                    "response.failed",
                    "response.incomplete",
                ):
                    event = candidate
                    break
    details = error_details(event)
    code = details.get("code") or details.get("type")
    param = details.get("param")
    return (
        str(code)[:128] if code is not None else None,
        str(param)[:256] if param is not None else None,
    )


def install():
    """Only normalize transport metadata; use the official bounded recovery."""
    from roboicl.policy import bounded_policy

    bounded_policy._http_error_details = http_error_details
    for name in (
        "_provider_error_is_transient",
        "_provider_error_is_deterministic",
        "_provider_error_code",
    ):
        original = getattr(bounded_policy, name)
        if getattr(original, "_gateway_error_envelope", False):
            continue

        @wraps(original)
        def normalize(details, *args, _original=original):
            return _original(error_details(details), *args)

        normalize._gateway_error_envelope = True
        setattr(bounded_policy, name, normalize)

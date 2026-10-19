"""Synthetic gateway envelopes through the inherited RoboICL request/recovery path."""

import copy
import io
import json
import urllib.error
from dataclasses import replace
from pathlib import Path
import pytest
from tactalent.roboicl.policy import Model
from tactalent.roboicl.benchmarks import binding, task_context, task_prompt
from tactalent.roboicl.demonstrations import plan
from tactalent.roboicl.observations import input_contract
from tactalent.roboicl.upstream import protocol_sha256
from tactalent.roboicl.provider_errors import error_details, http_error_details, install
from roboicl.policy import bounded_policy
from roboicl.policy.dialogue_policy import Model as Official

FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures/roboicl_gateway_wrapped_errors.json").read_text()
)


@pytest.fixture
def model(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRA_API_KEY", "offline-only")
    monkeypatch.setenv("ASTRA_API_ENDPOINT", "https://offline.invalid/v1/responses")
    monkeypatch.setenv("ASTRA_STREAM", "1")
    monkeypatch.setattr(
        "tactalent.roboicl.policy.validate_reference_demonstrations", lambda ref, spec: plan(spec)
    )
    spec = binding()
    reference = dict(
        demonstration_contract=plan(spec),
        horizon=10,
        frequency=60,
        icl_mode="vision",
        input_contract=input_contract(spec, "vision"),
        messages=[],
        protocol_sha256=protocol_sha256(spec),
        instruction="task",
        task_context=task_context(spec),
        task_prompt=task_prompt(spec, "task"),
    )
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(reference))
    m = Model(path, tmp_path, icl_mode="vision")
    m.stable_prefix_hash = "offline-test"
    m.returned_reasoning_items = [
        {"type": "reasoning", "id": "rs_offline", "encrypted_content": "OPAQUE", "summary": []}
    ]
    return m


def payload(model, reasoning=True):
    context = [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "TRAIN and current observation"}],
        },
        {"type": "function_call", "name": "act", "call_id": "already_executed", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "already_executed", "output": "completed once"},
    ]
    return {
        "model": "gpt-6-astra",
        "instructions": "unchanged task",
        "input": copy.deepcopy(model.returned_reasoning_items if reasoning else []) + context,
        "tools": [],
        "reasoning": {"effort": "xhigh"},
    }


class Reply(io.BytesIO):
    def __init__(self, data, content_type="application/json"):
        super().__init__(data)
        self.headers = {"Content-Type": content_type}
        self.status = 200


def completed():
    return Reply(
        json.dumps(
            {
                "id": "offline-response",
                "status": "completed",
                "output": [],
                "usage": {"input_tokens": 11, "output_tokens": 3},
            }
        ).encode()
    )


def fake_http(monkeypatch, body, *, status=400, stream=False, repeat=False):
    calls = []

    def send(request, timeout=None):
        calls.append(json.loads(request.data))
        if len(calls) == 1 or repeat:
            if stream:
                return Reply(body.encode(), "text/event-stream")
            raise urllib.error.HTTPError(
                request.full_url, status, "offline fixture", {}, io.BytesIO(body.encode())
            )
        return completed()

    monkeypatch.setattr("urllib.request.urlopen", send)
    return calls


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda f: str(f["seed"]))
@pytest.mark.parametrize("stream", [False, True], ids=["http400-sse", "http200-sse"])
def test_wrapped_gateway_error_recovers_current_request(model, monkeypatch, fixture, stream):
    before = payload(model)
    calls = fake_http(monkeypatch, fixture["body"], stream=stream)
    result = model._request(before)
    assert result["status"] == "completed" and len(calls) == 2
    assert model.reasoning_replay_disabled is True
    expected = copy.deepcopy(calls[0])
    expected["input"] = [item for item in expected["input"] if item.get("type") != "reasoning"]
    assert calls[1] == expected  # All images, text, tools, action receipts and settings unchanged.
    assert before == payload(model)  # No in-place mutation of the caller's pending request.
    assert model.reported_tokens == 14
    assert model.pending_reserved_tokens > 0  # Unknown first-request cost stays reserved.
    assert model._request.__func__ is Official._request  # No controller/action retry wrapper.
    assert model.local_safety_holds == 0
    assert not model.ledger
    assert model.last_active_payload["input"] == calls[1]["input"]


def test_json_message_envelope_uses_same_recovery(model, monkeypatch):
    body = json.dumps(
        {
            "error": {
                "type": "invalid_request_error",
                "message": json.dumps(
                    {
                        "error": {
                            "code": "invalid_encrypted_content",
                            "type": "invalid_request_error",
                        }
                    }
                ),
            }
        }
    )
    calls = fake_http(monkeypatch, body)
    assert model._request(payload(model))["status"] == "completed"
    assert len(calls) == 2


def test_repeated_rejection_stops_after_one_repaired_retry(model, monkeypatch):
    calls = fake_http(monkeypatch, FIXTURES[0]["body"], repeat=True)
    with pytest.raises(bounded_policy.RequestTransportError):
        model._request(payload(model))
    assert len(calls) == 2
    assert model.reported_tokens == 0 and model.pending_reserved_tokens > 0
    assert not model.ledger and model.local_safety_holds == 0


def test_retry_limit_of_one_is_respected(model, monkeypatch):
    model.dialogue_config = replace(model.dialogue_config, max_transport_attempts=1)
    calls = fake_http(monkeypatch, FIXTURES[0]["body"])
    with pytest.raises(bounded_policy.RequestTransportError):
        model._request(payload(model))
    assert len(calls) == 1


@pytest.mark.parametrize(
    "status,body",
    [
        (401, json.dumps({"error": {"code": "invalid_api_key"}})),
        (400, json.dumps({"error": {"code": "insufficient_quota"}})),
        (400, json.dumps({"error": {"code": "unsupported_value"}})),
        (
            400,
            json.dumps(
                {
                    "error": {
                        "type": "invalid_request_error",
                        "message": "Words mentioning invalid_encrypted_content are not a code",
                    }
                }
            ),
        ),
    ],
)
def test_unrelated_errors_do_not_retry(model, monkeypatch, status, body):
    calls = fake_http(monkeypatch, body, status=status)
    with pytest.raises(bounded_policy.RequestTransportError):
        model._request(payload(model))
    assert len(calls) == 1


def test_missing_reasoning_does_not_retry_same_rejected_bytes(model, monkeypatch):
    model.returned_reasoning_items = []
    calls = fake_http(monkeypatch, FIXTURES[0]["body"])
    with pytest.raises(bounded_policy.RequestTransportError):
        model._request(payload(model, reasoning=False))
    assert len(calls) == 1


def test_decoder_is_bounded_and_keeps_explicit_codes():
    assert http_error_details(FIXTURES[0]["body"]) == ("invalid_encrypted_content", None)
    for text in (
        "not JSON",
        "data: " + json.dumps({"type": "error", "error": {"code": "invalid_encrypted_content"}}),
        "x" * 8193,
    ):
        assert http_error_details(text) == (None, None)
    assert (
        error_details(
            {
                "code": "insufficient_quota",
                "message": json.dumps({"error": {"code": "invalid_encrypted_content"}}),
            }
        )["code"]
        == "insufficient_quota"
    )
    assert error_details({"message": "invalid_encrypted_content"}) == {
        "message": "invalid_encrypted_content"
    }


def test_install_is_idempotent_and_does_not_override_dialogue():
    before = bounded_policy._provider_error_code
    install()
    assert bounded_policy._provider_error_code is before
    assert Model._request is Official._request

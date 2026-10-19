"""Single-arm/camera binding; official dialogue, retries, receipts and memory are inherited."""

import base64, copy, hashlib, json, os, time, urllib.request
from pathlib import Path
from urllib.parse import urlsplit
from io import BytesIO
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from .upstream import VENDOR, protocol_sha256
from .benchmarks import (
    binding,
    profile,
    validate_binding,
    validate_reference_horizon,
    task_prompt,
    validate_reference_task_context,
)
from .reference import observation
from .demonstrations import validate_reference_demonstrations
from .observations import (
    resolve_mode,
    validate_reference_inputs,
    allowed_panel_sizes,
    image_guidance,
)
from roboicl.policy import dialogue_policy as official
from roboicl.policy.bounded_policy import BudgetExceeded
from .provider_errors import install as install_provider_error_compatibility

install_provider_error_compatibility()
# Robot camera whitelist: the released validator reads this module constant.
official.CAMERAS = ("head", "wrist")
from roboicl.policy import image_budget

_official_image_count = image_budget._observation_image_count


def _native_panel_image_count(cameras, image_layout, *, name="cameras"):
    if tuple(cameras) in (("head",), ("head", "wrist")) and image_layout == "triptych":
        return 1
    return _official_image_count(cameras, image_layout, name=name)


image_budget._observation_image_count = _native_panel_image_count


class Model(official.Model):
    def __init__(self, reference, output, icl_mode=None):
        self.reference = json.loads(Path(reference).read_text())
        self.spec = validate_binding(self.reference.get("binding", binding()))
        self.cameras = tuple(self.spec["cameras"])
        official.CAMERAS = self.cameras
        if self.reference.get("protocol_sha256") != protocol_sha256(self.spec):
            raise ValueError("Reference/binding version mismatch; prepare again")
        h = validate_reference_horizon(self.reference, self.spec)
        validate_reference_task_context(self.reference, self.spec)
        self.icl_mode = resolve_mode(self.spec, icl_mode)
        validate_reference_inputs(self.reference, self.spec, self.icl_mode)
        demos = validate_reference_demonstrations(self.reference, self.spec)
        settings = json.loads((VENDOR / "configs/shared_harness.json").read_text())
        settings.update(
            json.loads((VENDOR / "configs/protocols/one_shot_j12_b12.json").read_text())[
                "harness_overrides"
            ]
        )
        settings.update(
            predict_horizon=h,
            execute_horizon=h,
            feedback_cameras=list(self.cameras),
            demo_cameras=list(self.cameras),
            include_head_calibration=True,
            live_anchor_count=demos["live_anchor_count"],
        )
        # One native sensor panel per observation; official image/history counts are retained.
        harness = Path(output) / "harness.json"
        harness.write_text(json.dumps(settings, indent=2))
        os.environ["ASTRA_HARNESS_CONFIG"] = str(harness)
        os.environ["ASTRA_REASONING_EFFORT"] = settings["reasoning_effort"]
        os.environ["ASTRA_LOG_DIR"] = str(Path(output) / "traces")
        os.environ.pop("ASTRA_DEMO_MANIFEST", None)
        super().__init__({"model": "gpt-6-astra", "endpoint": responses_endpoint()})
        prompt = self.base_prompt
        prompt = prompt.replace(
            official.GRIPPER_GUIDANCE,
            profile(self.spec["benchmark"]).gripper_guidance(official.GRIPPER_GUIDANCE),
        )
        prompt = prompt.replace("Control dual ARX X5", "Control a single Franka Panda arm")
        prompt = prompt.replace(
            "EE means link6, not fingertip/TCP.", "EE means panda_hand, not fingertip/TCP."
        )
        prompt = prompt.replace(
            official.ARM_WORKSPACE_GUIDANCE + " " + official.ARM_POSTURE_GUIDANCE + " ", ""
        )
        prompt = prompt.replace(
            "Each action row has 14 numbers: [left6,left_gripper,right6,right_gripper].",
            "Each action row has 7 numbers: [dx,dy,dz,rx,ry,rz,gripper].",
        )
        prompt = prompt.replace("each arm6=", "arm6=")
        prompt = prompt.replace(
            "Commands run at 25Hz.", f"Commands run at {self.reference['frequency']:g}Hz."
        )
        prompt = prompt.replace("neither arm nor gripper", "neither the arm nor gripper")
        prompt += "\n\n" + image_guidance(self.spec, self.icl_mode)
        from roboicl.policy.head_camera_prompt import calibration_conventions

        prompt += "\n\n" + calibration_conventions()["text"]
        self.prompt = self.base_prompt = prompt
        self.frozen_tools = self._tools()
        self.last_wire_response = None

    def _tools(self):
        result = super()._tools()
        result[0]["description"] = result[0]["description"].replace("[T,14]", "[T,7]")
        row = result[0]["parameters"]["properties"]["actions"]["items"]
        row["minItems"] = row["maxItems"] = 7
        return result

    def _convert_tensor(self, rows, state):
        c = self.dialogue_config
        if not isinstance(rows, list) or len(rows) != c.predict_horizon:
            raise ValueError(f"actions must contain exactly {c.predict_horizon} steps")
        if any(
            not isinstance(r, list) or len(r) != 7 or any(type(x) not in (int, float) for x in r)
            for r in rows
        ):
            raise ValueError("actions must be numeric [T,7]")
        x = np.asarray(rows, float)
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite actions")
        if (
            np.max(np.linalg.norm(x[:, :3], axis=1)) > c.max_translation_m + 1e-6
            or np.max(np.linalg.norm(x[:, 3:6], axis=1)) > c.max_rotation_rad + 1e-6
        ):
            raise ValueError("Translation/rotation norm exceeds the official per-step limit")
        if np.any((x[:, 6] < 0) | (x[:, 6] > 1)):
            raise ValueError("Gripper must be in [0,1]")
        p = np.asarray(state["ee_pose"][:3], float)
        q = Rotation.from_quat(np.array(state["ee_pose"][3:])[[1, 2, 3, 0]])
        result = []
        for row in x:
            p = p + row[:3]
            q = Rotation.from_rotvec(row[3:6]) * q
            result.append(
                {
                    "mode": "ee",
                    "ee_pose": [*p.tolist(), *q.as_quat()[[3, 0, 1, 2]].tolist()],
                    "gripper": float(row[6]),
                }
            )
        return result

    def update_native(
        self, state, images, step, instruction, maximum, executed=None, tracking=None
    ):
        if step in self.archive:
            raise ValueError("Duplicate observation")
        obs = observation(state, images, step, "LIVE", self.cameras, self.icl_mode, self.spec)
        self.archive[step] = {
            "step": step,
            "state": state,
            "instruction": instruction,
            "message": obs,
            "max_episode_steps": maximum,
        }
        self.step = step
        if executed is not None:
            self.ledger.append(
                {
                    "resulting_step": step,
                    "executed_action": executed,
                    "state_after": state,
                    "tracking": tracking or {},
                }
            )
        self._log("observations.jsonl", self.archive[step])

    def _observations(self, steps, cameras=None):
        return [
            part
            for step in steps
            for part in copy.deepcopy(self.archive[step]["message"]["content"])
        ]

    def _initialize(self):
        from .calibration import authored, message

        active_task = {
            "type": "input_text",
            "text": task_prompt(self.spec, self.archive[0]["instruction"]),
        }
        live_header = [message(self.live_calibration, "TEST_RUNTIME")]
        if self.reference["demonstration_contract"]["shots"] > 1:
            live_header.insert(0, copy.deepcopy(active_task))
        self.messages = (
            [
                {
                    "role": "user",
                    "content": [
                        active_task,
                        message(
                            self.reference.get(
                                "training_calibration", authored(self.spec, training=True)
                            ),
                            "TRAIN",
                        ),
                    ],
                }
            ]
            + copy.deepcopy(self.reference["messages"])
            + [{"role": "user", "content": live_header}]
        )
        self.seen_call_ids.update(
            m["call_id"] for m in self.messages if m.get("type") == "function_call"
        )
        self.fixed_prefix_length = len(self.messages)
        self.stable_prefix_hash = hashlib.sha256(
            json.dumps(
                {
                    "model": self.model_name,
                    "instructions": self.prompt,
                    "tools": self.frozen_tools,
                    "prefix": self.messages,
                    "effort": self.effort,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        self.messages.append({"role": "user", "content": self._observations([0])})
        self.live_memory.configure(self.archive[0]["max_episode_steps"])
        self.live_history_tail_start = len(self.messages)

    def payload_metrics(self, payload):
        images = []

        def scrub(value):
            if isinstance(value, dict):
                if value.get("type") == "input_image":
                    if value.get("detail") != "original":
                        raise ValueError("RoboICL requires original image detail")
                    with Image.open(
                        BytesIO(base64.b64decode(value["image_url"].split(",", 1)[1]))
                    ) as pic:
                        valid_sizes = allowed_panel_sizes(self.spec, self.icl_mode)
                        if pic.size not in valid_sizes or pic.mode != "RGB":
                            raise ValueError("Invalid task-authorized native camera panel")
                    images.append(value)
                    return {"type": "input_image", "detail": "original"}
                return {k: scrub(v) for k, v in value.items()}
            if isinstance(value, list):
                return [scrub(v) for v in value]
            return value

        text_bytes = len(json.dumps(scrub(payload)).encode())
        wire_bytes = len(json.dumps(payload).encode())
        if len(images) > self.dialogue_config.max_context_images:
            raise BudgetExceeded("Image cap reached")
        return {
            "text_bytes": text_bytes,
            "image_count": len(images),
            "wire_bytes": wire_bytes,
            "reserved_tokens": text_bytes + len(images) * 1024 + self.max_output,
            "reservation_note": "Official budget estimate; never a bill",
        }

    def _observed_state_safety_hold(self, reason):
        maximum = self.archive[self.step]["max_episode_steps"]
        if not 0 <= self.step < maximum:
            return None
        state = self.archive[self.step]["state"]
        self.local_safety_holds += 1
        self.pending = {
            "start": self.step,
            "predicted": 1,
            "returned": 1,
            "execution_note": "Local observed-state safety hold.",
            "local_safety_hold": {"reason": str(reason)[:500], "index": self.local_safety_holds},
        }
        return [
            {
                "mode": "joint",
                "arm": state["arm_joint_state"],
                "gripper": state["ee_joint_state"][0],
            }
        ]


def responses_endpoint():
    endpoint = os.environ.get("ASTRA_API_ENDPOINT", "").strip()
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or not parsed.path.rstrip("/").endswith("/responses")
    ):
        raise ValueError("Set ASTRA_API_ENDPOINT to a credential-free HTTPS Responses endpoint")
    return endpoint.rstrip("/")


class ProxyHTTPS(urllib.request.HTTPSHandler):
    """Optional loopback CONNECT relay for the explicitly configured API endpoint."""

    def https_open(self, request):
        proxy = os.environ.get("ASTRA_API_PROXY_URL")
        if proxy and request.host == urlsplit(responses_endpoint()).netloc:
            from tactalent.utils.api import validate_api_proxy_url

            validated = validate_api_proxy_url(proxy)
            request.add_unredirected_header("Host", request.host)
            request.set_proxy(urlsplit(validated).netloc, "http")
        return super().https_open(request)

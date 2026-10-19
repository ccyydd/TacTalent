"""Explicit, symmetric TRAIN/LIVE sensor modes without spatial resampling."""

import base64
from io import BytesIO
from pathlib import Path
import numpy as np
from PIL import Image
from .benchmarks import profile

ICL_MODES = ("vision", "vision_tactile")
TACTILE_NAMES = ("tactile_left", "tactile_right")


def resolve_mode(spec, mode=None):
    mode = profile(spec["benchmark"]).DEFAULT_ICL_MODE if mode is None else mode
    if mode not in ICL_MODES:
        raise ValueError(f"Unknown ICL mode: {mode!r}")
    return mode


def image_names(spec, mode=None):
    mode = resolve_mode(spec, mode)
    return tuple(spec["cameras"]) + (TACTILE_NAMES if mode == "vision_tactile" else ())


def input_contract(spec, mode=None):
    mode = resolve_mode(spec, mode)
    return dict(
        version="native_sensor_rows_v1",
        icl_mode=mode,
        image_names=list(image_names(spec, mode)),
        tactile_field=(
            profile(spec["benchmark"]).TACTILE_FIELD if mode == "vision_tactile" else None
        ),
        crop=False,
        resize=False,
        flip=False,
        rotate=False,
    )


def validate_reference_inputs(reference, spec, mode=None):
    expected = input_contract(spec, mode)
    if (
        reference.get("input_contract") != expected
        or reference.get("icl_mode") != expected["icl_mode"]
    ):
        raise ValueError(
            "Reference ICL mode/sensor contract mismatch; prepare a matching reference"
        )
    return expected


def tactile_key(group, side):
    candidates = [key for key in (side + "_tactile", side + "_gsmini") if key in group]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one {side} RGB tactile sensor")
    return candidates[0]


def native_images(observation, spec, mode=None):
    from tactalent.benchmarks.cameras import _rgb

    mode = resolve_mode(spec, mode)
    images = {name: _rgb(observation["observation"][name]["rgb"], name) for name in spec["cameras"]}
    if mode == "vision_tactile":
        group = observation["tactile"]
        for side in ("left", "right"):
            key = tactile_key(group, side)
            name = "tactile_" + side
            images[name] = _rgb(group[key][profile(spec["benchmark"]).TACTILE_FIELD], name)
    return images


def panel(images, cameras, mode="vision", spec=None):
    cameras = tuple(cameras)
    if cameras not in (("head",), ("head", "wrist")) or mode not in ICL_MODES:
        raise ValueError("Unreviewed RGB cameras or ICL mode")
    names = cameras + (TACTILE_NAMES if mode == "vision_tactile" else ())
    if set(images) != set(names):
        raise ValueError("Images must exactly match the task-authorized ICL sensors")
    if mode == "vision_tactile" and spec is None:
        raise ValueError("Tactile panels require a benchmark profile")
    pictures = {}
    for name in names:
        source = images[name]
        if isinstance(source, (str, Path)):
            with Image.open(source) as image:
                image.load()
                picture = image.copy()
        else:
            value = np.asarray(source)
            if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
                raise ValueError("Expected uint8 native RGB")
            picture = Image.fromarray(value)
        sizes = (
            profile(spec["benchmark"]).TACTILE_SIZE_WH
            if name in TACTILE_NAMES
            else ((480, 270), (640, 360))
        )
        if picture.mode != "RGB" or picture.size not in sizes:
            raise ValueError(f"Unexpected native raster for {name}: {picture.size}, {picture.mode}")
        pictures[name] = picture
    if len({pictures[name].size for name in cameras}) != 1:
        raise ValueError("RGB camera rasters must match within one observation")
    if mode == "vision_tactile" and pictures["tactile_left"].size != pictures["tactile_right"].size:
        raise ValueError("Left/right tactile rasters must match")
    rows = [cameras] + ([TACTILE_NAMES] if mode == "vision_tactile" else [])
    width = max(sum(pictures[name].width for name in row) for row in rows)
    height = sum(max(pictures[name].height for name in row) for row in rows)
    canvas = Image.new("RGB", (width, height))
    layout = []
    y = 0
    for row in rows:
        x = 0
        for name in row:
            picture = pictures[name]
            canvas.paste(picture, (x, y))
            layout.append(dict(sensor=name, xywh=[x, y, picture.width, picture.height]))
            x += picture.width
        y += max(pictures[name].height for name in row)
    return canvas, layout


def encoded_panel(images, cameras, mode="vision", spec=None):
    canvas, layout = panel(images, cameras, mode, spec)
    out = BytesIO()
    canvas.save(out, format="JPEG", quality=90, subsampling=0)
    return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode(), layout


def allowed_panel_sizes(spec, mode=None):
    mode = resolve_mode(spec, mode)
    rgb = [(w * len(spec["cameras"]), h) for w, h in ((480, 270), (640, 360))]
    if mode == "vision":
        return set(rgb)
    return {
        (max(w, 2 * tw), h + th)
        for w, h in rgb
        for tw, th in profile(spec["benchmark"]).TACTILE_SIZE_WH
    }


def image_guidance(spec, mode=None):
    owner = profile(spec["benchmark"])
    mode = resolve_mode(spec, mode)
    cameras = "head only" if spec["cameras"] == ["head"] else "head | wrist (left | right)"
    template = owner.VISION_IMAGE_GUIDANCE if mode == "vision" else owner.TACTILE_IMAGE_GUIDANCE
    return template.format(cameras=cameras)

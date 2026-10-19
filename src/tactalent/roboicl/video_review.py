"""Update the persistent RoboICL review UI from immutable episode artifacts.

This renderer changes display copies only. It never changes model input images,
evaluation results, demonstrations or simulator settings.
"""

import argparse
import fcntl
from contextlib import contextmanager
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import uuid
from datetime import datetime, timezone

from PIL import Image, ImageDraw, ImageFont, ImageOps
from tactalent.utils.video import probe

from tactalent.paths import workspace_root
DEFAULT_OUTPUT = workspace_root() / "reports/roboicl_review"
# Match UniVTAC envs/utils/data.py:VideoHandler.reset export cadence.
DISPLAY_FPS = 10
RENDER_VERSION = "matched_sensor_rows_v2_10fps"
ORDER = {
    "UniVTAC": [
        "lift_bottle",
        "pull_out_key",
        "lift_can",
        "put_bottle_in_shelf",
        "insert_hole",
        "insert_HDMI",
        "insert_tube",
        "grasp_classify",
    ],
    "OpenViTac": [
        "can_empty_select",
        "weight_classify",
        "hardness_classify",
        "roughness_classify",
        "roughness_regrasp",
        "grasp_fragile_chip",
        "turn_gear_pair",
        "pull_drawer",
        "insert_USB",
        "insert_block_v1",
        "insert_block_v2",
    ],
}
NAMES = {
    "lift_bottle": "Lift Bottle",
    "pull_out_key": "Pull-out Key",
    "lift_can": "Lift Can",
    "put_bottle_in_shelf": "Put Bottle",
    "insert_hole": "Insert Hole",
    "insert_HDMI": "Insert HDMI",
    "insert_tube": "Insert Tube",
    "grasp_classify": "Grasp Classify",
    "can_empty_select": "Can Empty Select",
    "weight_classify": "Weight Classify",
    "hardness_classify": "Hardness Classify",
    "roughness_classify": "Roughness Classify",
    "roughness_regrasp": "Roughness Regrasp",
    "grasp_fragile_chip": "Grasp Fragile Chip",
    "turn_gear_pair": "Turn Gear Pair",
    "pull_drawer": "Pull Drawer",
    "insert_USB": "Insert USB",
    "insert_block_v1": "Insert Block v1",
    "insert_block_v2": "Insert Block v2",
}
LABELS = {
    "head": "Fixed camera",
    "wrist": "Wrist camera",
    "tactile_left": "Left tactile",
    "tactile_right": "Right tactile",
}


def read(path):
    return json.loads(Path(path).read_text())


def atomic_write(path, text):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    tmp.write_text(text)
    os.replace(tmp, path)


def authorized_names(runtime):
    names = list(runtime["model_images"])
    expected = list(runtime["model_cameras"])
    if runtime["icl_mode"] == "vision_tactile":
        expected += ["tactile_left", "tactile_right"]
    if names != expected or expected[: len(runtime["model_cameras"])] not in (
        ["head"],
        ["head", "wrist"],
    ):
        raise ValueError("Unrecognized task sensor contract")
    return names


def display_layout(names):
    rows = [
        [k for k in ("head", "wrist") if k in names],
        [k for k in ("tactile_left", "tactile_right") if k in names],
    ]
    rows = [row for row in rows if row]
    if {n for row in rows for n in row} != set(names) or not rows:
        raise ValueError("Unsupported review sensor names")
    sizes = {n: (320, 240) if n.startswith("tactile") else (480, 270) for n in names}
    width = max(sum(sizes[n][0] for n in row) for row in rows)
    y, layout = 0, {}
    for row in rows:
        x = (width - sum(sizes[n][0] for n in row)) // 2
        for name in row:
            w, h = sizes[name]
            layout[name] = (x, y + 24, w, h)
            x += w
        y += max(sizes[n][1] for n in row) + 24
    return (width, y), layout


def render_panel(pictures, names):
    """Full views only: aspect-preserving fit, identical tile geometry and labels."""
    canvas_size, layout = display_layout(names)
    canvas = Image.new("RGB", canvas_size)
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    for name, (x, y, w, h) in layout.items():
        picture = pictures[name]
        fitted = ImageOps.contain(picture.convert("RGB"), (w, h), Image.Resampling.LANCZOS)
        canvas.paste(fitted, (x + (w - fitted.width) // 2, y + (h - fitted.height) // 2))
        draw.text((x + 5, y - 21), LABELS[name], font=font, fill="#c6d5e8")
    return canvas


def native_rectangles(runtime, stream):
    """Recover complete views from the existing centered native review montage.

    These rectangles match presentation_task_class; they are not image-content
    crops. Unexpected native montage geometry fails instead of guessing.
    """
    sizes = runtime["native_source_sizes_wh"]
    hw, hh = sizes["head"]
    tw, th = sizes.get(
        "tactile_left", [400, 300] if runtime["binding"]["benchmark"] == "UniVTAC" else [320, 240]
    )
    width, height = stream["width"], stream["height"]
    if (width, height) != (max(hw * 2, 800), hh + th):
        raise ValueError(f"Unknown native montage: {(width, height)} for {sizes}")
    x = (width - hw * 2) // 2
    tx = (width - tw * 2) // 2
    return {
        "head": (x, 0, hw, hh),
        "wrist": (x + hw, 0, hw, hh),
        "tactile_left": (tx, hh, tw, th),
        "tactile_right": (tx + tw, hh, tw, th),
    }


def demo_frames(source, names):
    for obs in read(source)["observations"]:
        pictures = {}
        for name in names:
            with Image.open(obs["images"][name]) as picture:
                pictures[name] = picture.copy()
        yield render_panel(pictures, names)


def evaluation_frames(source, runtime, stream, names):
    boxes = native_rectangles(runtime, stream)
    width, height = stream["width"], stream["height"]
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-threads",
            "2",
            "-i",
            str(source),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
    )
    try:
        while True:
            data = process.stdout.read(width * height * 3)
            if not data:
                break
            if len(data) != width * height * 3:
                raise ValueError("Truncated decoded video frame")
            frame = Image.frombytes("RGB", (width, height), data)
            pictures = {
                name: frame.crop((x, y, x + w, y + h))
                for name in names
                for x, y, w, h in [boxes[name]]
            }
            yield render_panel(pictures, names)
        if process.wait() != 0:
            raise RuntimeError("Native review video decode failed")
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.kill()
            process.wait()


@contextmanager
def exclusive_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def encode(source, output, kind, names, runtime=None):
    # Few-shot variants can share a recording and reach the cache concurrently.
    key = hashlib.sha256((str(source) + kind + repr(names)).encode()).hexdigest()
    with exclusive_lock(output / "media" / (key + ".lock")):
        return _encode(source, output, kind, names, runtime)


def _encode(source, output, kind, names, runtime=None):
    source = Path(source).resolve(strict=True)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    settings = {
        "version": RENDER_VERSION,
        "kind": kind,
        "names": names,
        "fps": DISPLAY_FPS,
        "source_sha256": source_sha,
    }
    if runtime is not None:
        settings["source_sizes_wh"] = runtime["native_source_sizes_wh"]
    cache_key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    dest = output / "media" / (cache_key[:24] + ".mp4")
    poster, receipt_path = dest.with_suffix(".jpg"), dest.with_suffix(".json")
    if dest.exists() and poster.exists() and receipt_path.exists():
        receipt = read(receipt_path)
        if receipt["cache_key"] == cache_key:
            return receipt
    dest.parent.mkdir(parents=True, exist_ok=True)
    if kind == "demo":
        recording = read(source)
        source_frames = len(recording["observations"])
        source_fps = 1 / float(recording["metadata"]["control_dt"])
        source_sizes = {}
        for n in names:
            with Image.open(recording["observations"][0]["images"][n]) as image:
                source_sizes[n] = list(image.size)
        frames = demo_frames(source, names)
        boxes = None
    else:
        stream = probe(source)
        source_frames, source_fps = int(stream["nb_frames"]), stream["r_frame_rate"]
        source_sizes = {n: runtime["native_source_sizes_wh"][n] for n in names}
        boxes = {n: list(native_rectangles(runtime, stream)[n]) for n in names}
        frames = evaluation_frames(source, runtime, stream, names)
    width, height = display_layout(names)[0]
    temp = dest.with_suffix(".part.mp4")
    encoder = subprocess.Popen(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(DISPLAY_FPS),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "2",
            "-movflags",
            "+faststart",
            str(temp),
        ],
        stdin=subprocess.PIPE,
    )
    count = 0
    try:
        for frame in frames:
            if count == 0:
                frame.save(poster, quality=94)
            encoder.stdin.write(frame.tobytes())
            count += 1
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("Review encode failed")
    except BaseException:
        encoder.kill()
        encoder.wait()
        raise
    finally:
        frames.close()
    stream = probe(temp)
    assert count == source_frames == int(stream["nb_frames"])
    assert stream["r_frame_rate"] == f"{DISPLAY_FPS}/1"
    assert (stream["width"], stream["height"]) == (width, height)
    assert abs(float(stream["duration"]) - count / DISPLAY_FPS) < 0.001
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-threads", "2", "-i", str(temp), "-f", "null", "-"],
        check=True,
    )
    temp.replace(dest)
    receipt = dict(
        cache_key=cache_key,
        source=str(source),
        source_sha256=source_sha,
        src=str(dest.relative_to(output)),
        poster=str(poster.relative_to(output)),
        video=stream,
        source_fps=source_fps,
        source_frame_count=source_frames,
        source_sizes_wh=source_sizes,
        displayed_sensors=names,
        source_rectangles=boxes,
        image_transform="Whole views; aspect-preserving fit into common tiles, padding and identical labels. No flip or content crop.",
        cadence="All recorded frames retained; 10 control frames per playback second; display only.",
    )
    atomic_write(receipt_path, json.dumps(receipt, indent=2))
    print("READY", kind, source.name, count, "frames", flush=True)
    return receipt


def result_note(result):
    if result["status"] == "error":
        return "运行异常，不能计为策略失败。详情见结束原因和原始结果记录。"
    rules = result.get("evaluation_rules", {})
    modified = rules.get("matches_upstream_protocol") is False
    prefix = ("修改协议：" + rules["protocol"] + "；") if modified else ""
    diagnostic = result.get("native_diagnostics", {}).get("success_diagnostics", {})
    reason = diagnostic.get("terminal_reason")
    if reason:
        note = prefix + ("结束原因：" if modified else "官方结束原因：") + reason + "。"
        if reason == "early_release":
            note += "松手前尚未验证稳定支撑。"
        if diagnostic.get("last_physical_sample", {}).get("fractured") is False:
            note += "本集未检测到物体破碎。"
        return note
    if result.get("termination") == "native_horizon":
        return prefix + "达到官方动作步数上限。"
    return prefix


def job_from_episode(output):
    output = Path(output).resolve(strict=True)
    result = read(output / "result.json")
    task_id = result["benchmark"] + "_" + result["task"]
    if result.get("variant"):
        task_id += "_" + result["variant"]
    return dict(
        job_id=task_id,
        benchmark=result["benchmark"],
        task=result["task"],
        variant=result.get("variant"),
        seed=result["seed"],
        output=str(output),
        result=result,
    )


def build_task(job, output, *, include_evaluations=True):
    folder = Path(job["output"])
    result, runtime, ref = (
        read(folder / "result.json"),
        read(folder / "runtime.json"),
        read(folder / "reference.snapshot.json"),
    )
    names = authorized_names(runtime)
    task = dict(
        id=job["job_id"],
        task=job["task"],
        name=NAMES[job["task"]],
        benchmark=job["benchmark"],
        variant=job.get("variant"),
        status=result["status"],
        steps=result.get("native_steps"),
        decisions=result.get("decisions"),
        seed=job["seed"],
        output=str(folder),
        note=result_note(result),
        model_images=names,
        horizon=ref["horizon"],
        native_horizon=runtime["native_horizon"],
        frequency=runtime["control_frequency_hz"],
        demos=[],
        evaluations=[],
        comparison_outputs=list(job.get("comparison_outputs", [])),
    )
    for index, demo in enumerate(ref["demonstrations"]):
        item = encode(demo["source"], output, "demo", names)
        item.update(
            label=f"示范 {index+1}"
            + (" · " + ", ".join(demo["coverage"]) if demo.get("coverage") else ""),
            seed=demo.get("seed"),
            episode_id=demo.get("episode_id"),
            starts=demo.get("starts", ref["starts"]),
            horizon=demo.get("horizon", ref["horizon"]),
            coverage=demo.get("coverage", []),
        )
        task["demos"].append(item)
    if not include_evaluations:
        return task
    rerun_label = ""
    if job.get("previous_output"):
        previous_status = read(Path(job["previous_output"]) / "result.json")["status"]
        rerun_label = "（异常重跑）" if previous_status == "error" else "（重跑）"
    outputs = [("最新评测" + rerun_label, folder)]
    if job.get("previous_output"):
        outputs.append(("重跑前记录", Path(job["previous_output"])))
    seen = {str(path.resolve()) for _, path in outputs}
    for comparison in task["comparison_outputs"]:
        path = Path(comparison).resolve(strict=True)
        if str(path) not in seen:
            outputs.append(("对照评测", path))
            seen.add(str(path))
    for label, path in outputs:
        current = read(path / "result.json")
        if (current["benchmark"], current["task"], current.get("variant")) != (
            result["benchmark"],
            result["task"],
            result.get("variant"),
        ):
            raise ValueError("Comparison episodes must belong to the same task and variant")
        conditions = current.get("evaluation_case", {}).get("conditions", {})
        case_label = ", ".join(str(v) for v in conditions.values())
        label += (f" · {case_label}" if case_label else "") + f" · seed {current['seed']}"
        eval_runtime = read(path / "runtime.json")
        if authorized_names(eval_runtime) != names:
            raise ValueError("Cannot pair evaluation attempts with different model sensors")
        videos = list((path / "native/video").rglob("*.mp4"))
        if len(videos) != 1:
            raise ValueError(f"Expected one complete native review video: {path}")
        item = encode(videos[0], output, "eval", names, eval_runtime)
        current = read(path / "result.json")
        item.update(
            label=label,
            output=str(path),
            status=current["status"],
            steps=current.get("native_steps"),
            termination=current.get("termination"),
            note=result_note(current),
        )
        task["evaluations"].append(item)
    return task


def merge_tasks(existing, incoming):
    merged = {t["id"]: t for t in existing}
    for task in incoming:
        merged[task["id"]] = task
    return sorted(
        merged.values(),
        key=lambda t: (
            list(ORDER).index(t["benchmark"]),
            ORDER[t["benchmark"]].index(t.get("task", t["id"].removeprefix(t["benchmark"] + "_"))),
            t.get("variant") or "",
        ),
    )


def build(jobs, output=DEFAULT_OUTPUT):
    from tactalent.utils.storage import require_output_path

    output = require_output_path(output).resolve()
    with exclusive_lock(output / ".build.lock"):
        return _build(jobs, output)


def _build(jobs, output):
    from tactalent.utils.storage import require_output_path

    output = require_output_path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    old = read(output / "manifest.json") if (output / "manifest.json").exists() else {}
    if len({j["job_id"] for j in jobs}) != len(jobs):
        raise ValueError("Choose one latest episode per task/variant in each update")
    previous_tasks = {task["id"]: task for task in old.get("tasks", [])}
    # Explicitly pinned comparisons survive automatic latest-episode updates.
    jobs = [
        dict(
            job,
            comparison_outputs=list(
                dict.fromkeys(
                    previous_tasks.get(job["job_id"], {}).get("comparison_outputs", [])
                    + job.get("comparison_outputs", [])
                )
            ),
        )
        for job in jobs
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        incoming = list(pool.map(lambda job: build_task(job, output), jobs))
    tasks = merge_tasks(old.get("tasks", []), incoming)
    manifest = dict(
        title="RoboICL · 全任务视频对照",
        renderer=RENDER_VERSION,
        updated_utc=datetime.now(timezone.utc).isoformat(),
        tasks=tasks,
        counts={s: sum(t["status"] == s for t in tasks) for s in ("success", "failed", "error")},
        demo_count=sum(len(t["demos"]) for t in tasks),
        evaluation_count=sum(len(t["evaluations"]) for t in tasks),
        display_fps=DISPLAY_FPS,
    )
    page = (Path(__file__).parent / "templates/review.html").read_text()
    page = page.replace("__DATA__", json.dumps(manifest, ensure_ascii=False).replace("</", "<\\/"))
    # Content-addressed media and atomic page replacement keep the previous UI
    # usable until all updated videos have encoded and passed full decode checks.
    if old:
        history = output / "history"
        history.mkdir(exist_ok=True)
        atomic_write(history / (str(time_ns()) + ".json"), json.dumps(old, ensure_ascii=False))
    atomic_write(output / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    atomic_write(output / "index.html", page)
    return manifest


def time_ns():
    import time

    return time.time_ns()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, action="append", default=[])
    parser.add_argument("--episode", type=Path, action="append", default=[])
    parser.add_argument(
        "--compare-episode",
        type=Path,
        action="append",
        default=[],
        help="Keep this episode available alongside later results for its task",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    jobs = [job for source in args.summary for job in read(source)["episodes"]]
    jobs += [job_from_episode(path) for path in args.episode]
    if not jobs:
        parser.error("Provide --summary or --episode")
    for path in args.compare_episode:
        comparison = job_from_episode(path)
        matching = [job for job in jobs if job["job_id"] == comparison["job_id"]]
        if len(matching) != 1:
            parser.error("Each comparison needs exactly one selected episode from the same task")
        matching[0].setdefault("comparison_outputs", []).append(comparison["output"])
    result = build(jobs, args.output)
    print(
        json.dumps(
            dict(
                tasks=len(result["tasks"]),
                demos=result["demo_count"],
                videos=result["evaluation_count"],
                counts=result["counts"],
                page=str(args.output.resolve() / "index.html"),
            )
        )
    )


if __name__ == "__main__":
    main()

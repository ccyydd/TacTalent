"""Publish every episode in a batch to the persistent review page.

Display-only: result files, reference banks and original videos are never edited.
"""

import argparse
from collections import Counter, defaultdict
import concurrent.futures
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from tactalent.roboicl import video_review as review


def episode_identity(row):
    return (
        row["benchmark"],
        row["task"],
        row.get("variant"),
        row["seed"],
        row.get("icl_mode"),
        row.get("evaluation_case"),
        json.dumps(row.get("conditions", {}), sort_keys=True),
    )


def episode_item(row, output, names, exclusions):
    folder = Path(row["output"])
    result = row["result"]
    status = "initialization_failed" if str(folder) in exclusions else result["status"]
    case = result.get("evaluation_case", {})
    conditions = row.get("conditions", case.get("conditions", {}) if isinstance(case, dict) else {})
    status_names = dict(
        success="成功", failed="任务失败", error="运行异常", initialization_failed="初始化失败"
    )
    label = " · ".join(
        [f"seed {row['seed']}"]
        + [str(v) for v in conditions.values()]
        + [status_names.get(status, status)]
    )
    item = dict(
        label=label,
        output=str(folder),
        seed=row["seed"],
        evaluation_case=row.get("evaluation_case"),
        conditions=conditions,
        icl_mode=row.get("icl_mode"),
        status=status,
        raw_status=result["status"],
        steps=result.get("native_steps"),
        decisions=result.get("decisions"),
        termination=result.get("termination"),
        note=review.result_note(result),
        available=False,
    )
    videos = list((folder / "native/video").rglob("*.mp4"))
    if not videos:
        item["unavailable_reason"] = (
            "初始化未完成，没有生成评测录像。"
            if status == "initialization_failed"
            else "运行未正常启动，没有生成评测录像。"
        )
        return item
    if len(videos) != 1:
        raise ValueError(f"Ambiguous native video sources: {folder}")
    check = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(videos[0])],
        capture_output=True,
        text=True,
        check=True,
    )
    streams = [s for s in json.loads(check.stdout)["streams"] if s["codec_type"] == "video"]
    item["source"] = str(videos[0])
    if not streams:
        item["unavailable_reason"] = (
            "初始化未完成；录像文件中没有图像帧。"
            if status == "initialization_failed"
            else "首个动作前发生接口异常；录像文件中没有图像帧。"
        )
        return item
    runtime = review.read(folder / "runtime.json")
    if review.authorized_names(runtime) != names:
        raise ValueError(f"Sensor mismatch in batch: {folder}")
    rendered = review.encode(videos[0], output, "eval", names, runtime)
    item.update(rendered, available=True)
    if status == "error":
        item["note"] = "运行中断前的部分录像；不计为策略失败。"
    return item


def reference_signature(folder):
    ref = review.read(Path(folder) / "reference.snapshot.json")
    return [
        (d["source"], d.get("starts", ref["starts"]), d.get("horizon", ref["horizon"]))
        for d in ref["demonstrations"]
    ]


def build_group(rows, output, exclusions):
    # Result-only launcher failures still get a visible record in the selector.
    complete = [
        r
        for r in rows
        if all(
            (Path(r["output"]) / n).exists()
            for n in ("result.json", "runtime.json", "reference.snapshot.json")
        )
    ]
    if not complete:
        raise ValueError("No model sensor/reference metadata for task " + rows[0]["task"])
    signatures = [reference_signature(r["output"]) for r in complete]
    if any(s != signatures[0] for s in signatures[1:]):
        raise ValueError("Different demonstration references need separate review groups")
    task = review.build_task(
        review.job_from_episode(complete[0]["output"]), output, include_evaluations=False
    )
    task["evaluations"] = [episode_item(r, output, task["model_images"], exclusions) for r in rows]
    task["comparison_outputs"] = []
    task["batch_counts"] = dict(Counter(e["status"] for e in task["evaluations"]))
    task["planned_episodes"] = len(rows)
    task["available_videos"] = sum(e["available"] for e in task["evaluations"])
    task["note"] = ""
    return task


def build_batch(summary, output=review.DEFAULT_OUTPUT, *, exclusions_path=None):
    from tactalent.utils.storage import require_output_path

    output = require_output_path(output).resolve()
    source = review.read(summary)
    rows = source["episodes"]
    identities = [episode_identity(r) for r in rows]
    if len(set(identities)) != len(rows) or len({r["output"] for r in rows}) != len(rows):
        raise ValueError("Duplicate episode identity or output in batch")
    exclusions = set()
    if exclusions_path:
        exclusions = {
            r["output"] for r in review.read(exclusions_path) if r.get("exclude_from_success_rate")
        }
    groups = defaultdict(list)
    for r in rows:
        groups[(r["benchmark"], r["task"], r.get("variant"), r.get("icl_mode"))].append(r)
    if len({r.get("icl_mode") for r in rows}) != 1:
        raise ValueError("Publish modalities separately so demonstration sensors match")
    with review.exclusive_lock(output / ".build.lock"):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            tasks = list(
                pool.map(lambda group: build_group(group, output, exclusions), groups.values())
            )
        tasks = review.merge_tasks([], tasks)
        all_items = [e for task in tasks for e in task["evaluations"]]
        counts = dict(Counter(e["status"] for e in all_items))
        manifest = dict(
            title="RoboICL · 全任务视频对照",
            renderer=review.RENDER_VERSION,
            updated_utc=datetime.now(timezone.utc).isoformat(),
            tasks=tasks,
            batch_source=str(Path(summary).resolve()),
            counts=counts,
            logical_task_count=len({(r["benchmark"], r["task"]) for r in rows}),
            episode_count=len(rows),
            evaluation_count=sum(e["available"] for e in all_items),
            unavailable_count=sum(not e["available"] for e in all_items),
            demo_count=sum(len(t["demos"]) for t in tasks),
            display_fps=review.DISPLAY_FPS,
        )
        old = output / "manifest.json"
        if old.exists():
            history = output / "history"
            history.mkdir(exist_ok=True)
            review.atomic_write(history / f"{review.time_ns()}.json", old.read_text())
        page = (Path(review.__file__).parent / "templates/review.html").read_text()
        page = page.replace(
            "__DATA__", json.dumps(manifest, ensure_ascii=False).replace("</", "<\\/")
        )
        review.atomic_write(
            output / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2)
        )
        review.atomic_write(output / "index.html", page)
        return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path)
    parser.add_argument("--output", type=Path, default=review.DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build_batch(args.summary, args.output, exclusions_path=args.exclusions)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("episode_count", "evaluation_count", "unavailable_count", "counts")
            }
        )
    )


if __name__ == "__main__":
    main()

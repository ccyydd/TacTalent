import json
import subprocess

import pytest
from PIL import Image

from tactalent.roboicl import video_review as review


def runtime(wrist=False, uni=False):
    names = ["head"] + (["wrist"] if wrist else []) + ["tactile_left", "tactile_right"]
    sizes = {
        "head": [640, 360] if uni else [480, 270],
        "tactile_left": [400, 300] if uni else [320, 240],
        "tactile_right": [400, 300] if uni else [320, 240],
    }
    if wrist:
        sizes["wrist"] = sizes["head"]
    return {
        "model_images": names,
        "model_cameras": names[:-2],
        "icl_mode": "vision_tactile",
        "native_source_sizes_wh": sizes,
        "binding": {"benchmark": "UniVTAC" if uni else "OpenViTac"},
    }


@pytest.mark.parametrize("uni", [False, True])
@pytest.mark.parametrize("wrist", [False, True])
def test_complete_sensor_views_match_demo_layout_without_unauthorized_wrist(tmp_path, uni, wrist):
    rt = runtime(wrist, uni)
    names = review.authorized_names(rt)
    hw, hh = rt["native_source_sizes_wh"]["head"]
    tw, th = rt["native_source_sizes_wh"]["tactile_left"]
    stream = {"width": hw * 2, "height": hh + th}
    boxes = review.native_rectangles(rt, stream)
    colors = {
        "head": (255, 0, 0),
        "wrist": (0, 255, 0),
        "tactile_left": (0, 0, 255),
        "tactile_right": (255, 255, 0),
    }
    native = Image.new("RGB", (stream["width"], stream["height"]))
    pictures = {}
    for name, (x, y, w, h) in boxes.items():
        im = Image.new("RGB", (w, h), colors[name])
        # Mark all four sensor borders to catch accidental image-content crops.
        from PIL import ImageDraw

        ImageDraw.Draw(im).rectangle((0, 0, w - 1, h - 1), outline="white", width=8)
        native.paste(im, (x, y))
        if name in names:
            pictures[name] = im
    src = tmp_path / "native.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{native.width}x{native.height}",
            "-r",
            "60",
            "-i",
            "pipe:0",
            "-frames:v",
            "3",
            "-c:v",
            "libx264rgb",
            "-crf",
            "0",
            "-threads",
            "1",
            str(src),
        ],
        input=native.tobytes() * 3,
        check=True,
    )
    expected = review.render_panel(pictures, names)
    actual = list(review.evaluation_frames(src, rt, stream, names))
    assert len(actual) == 3
    assert actual[0].tobytes() == expected.tobytes()
    if not wrist:
        assert (0, 255, 0) not in set(actual[0].getdata())
    else:
        assert (0, 255, 0) in set(actual[0].getdata())
    receipt = review.encode(src, tmp_path / "display", "eval", names, rt)
    assert receipt["video"]["r_frame_rate"] == "10/1"
    assert int(receipt["video"]["nb_frames"]) == 3
    assert float(receipt["video"]["duration"]) == 0.3
    assert receipt["displayed_sensors"] == names
    # Cache use must not alter the immutable source.
    assert review.encode(src, tmp_path / "display", "eval", names, rt) == receipt


def test_aspect_fit_preserves_complete_wide_image():
    im = Image.new("RGB", (800, 200), "red")
    result = review.render_panel({"head": im}, ["head"])
    # 4:1 input remains 480x120 inside the 480x270 tile.
    assert result.getpixel((0, 24)) == (0, 0, 0)
    assert result.getpixel((0, 99)) == (255, 0, 0)
    assert result.getpixel((479, 218)) == (255, 0, 0)
    assert result.getpixel((0, 219)) == (0, 0, 0)


def test_unknown_native_montage_is_rejected():
    with pytest.raises(ValueError, match="Unknown native montage"):
        review.native_rectangles(runtime(), {"width": 640, "height": 480})


def test_sensor_contract_disagreement_is_rejected():
    rt = runtime()
    rt["model_images"].insert(1, "wrist")
    with pytest.raises(ValueError, match="sensor contract"):
        review.authorized_names(rt)


def test_single_task_update_keeps_all_other_results():
    old = [
        {
            "id": "UniVTAC_lift_bottle",
            "benchmark": "UniVTAC",
            "task": "lift_bottle",
            "status": "failed",
        },
        {
            "id": "OpenViTac_turn_gear_pair",
            "benchmark": "OpenViTac",
            "task": "turn_gear_pair",
            "status": "success",
        },
    ]
    new = dict(old[0], status="success")
    assert review.merge_tasks(old, [new]) == [new, old[1]]
    assert old[0]["status"] == "failed"


def test_report_entry_point_routes_to_persistent_ui(tmp_path, monkeypatch):
    from tactalent.roboicl import report

    episode = tmp_path / "episode"
    episode.mkdir()
    (episode / "result.json").write_text(
        json.dumps(
            dict(
                benchmark="UniVTAC",
                task="pull_out_key",
                seed=1000000,
                status="failed",
                variant=None,
            )
        )
    )
    persistent = tmp_path / "persistent"
    calls = []
    monkeypatch.setattr(review, "DEFAULT_OUTPUT", persistent)
    monkeypatch.setattr(review, "build", lambda jobs: calls.append(jobs))
    assert report.build(episode) == str(persistent)
    assert calls[0][0]["job_id"] == "UniVTAC_pull_out_key"
    assert calls[0][0]["output"] == str(episode.resolve())


def test_modified_scoring_is_disclosed_in_review_note():
    result = dict(
        status="failed",
        native_diagnostics=dict(success_diagnostics=dict(terminal_reason="incomplete")),
    )
    assert review.result_note(result).startswith("官方结束原因")
    result["evaluation_rules"] = dict(
        protocol="roboicl_chip_safe_landing_no_overpress_v1", matches_upstream_protocol=False
    )
    note = review.result_note(result)
    assert "修改协议" in note and "roboicl_chip_safe_landing_no_overpress_v1" in note
    assert "官方结束原因" not in note

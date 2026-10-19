"""Update the review page for a completed evaluation episode."""

from pathlib import Path


def build(output):
    from .video_review import build as update, job_from_episode, DEFAULT_OUTPUT

    output = Path(output)
    if (output / "result.json").exists():
        update([job_from_episode(output)])
    return str(DEFAULT_OUTPUT)

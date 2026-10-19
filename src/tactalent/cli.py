"""The public TacTalent command line."""

import argparse
import runpy
import sys

COMMANDS = {
    "eval": ("Run one evaluation episode", "tactalent.roboicl.run"),
    "suite": ("Plan or run a stratified evaluation suite", "tactalent.roboicl.suite"),
    "prepare": ("Prepare demonstrations and reference catalogs", "tactalent.roboicl.prepare_bank"),
    "collect": (
        "Collect an OpenViTac scripted training episode",
        "tactalent.roboicl.collect_training",
    ),
    "review": ("Build a video review from saved episodes", "tactalent.roboicl.video_review"),
    "review-batch": ("Review saved evaluation batches", "tactalent.roboicl.review_batch"),
    "checkpoint": ("Continue or stop a paused simulator", "tactalent.roboicl.checkpoint"),
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="tactalent", description="RoboICL evaluation on UniVTAC and OpenViTac"
    )
    parser.add_argument("command", nargs="?", choices=["list", *COMMANDS], help="Action to perform")
    if not argv or argv[0] in ("-h", "--help"):
        parser.print_help()
        print("\nCommands:")
        print("  list          List the 19 supported tasks")
        for name, (description, _) in COMMANDS.items():
            print(f"  {name:13} {description}")
        return
    selected = parser.parse_args(argv[:1]).command
    if selected == "list":
        if argv[1:]:
            parser.error("list takes no arguments")
        from .roboicl.benchmarks import profile

        for benchmark in ("UniVTAC", "OpenViTac"):
            owner = profile(benchmark)
            print(f"{benchmark} ({len(owner.TASKS)} tasks)")
            for task in owner.TASKS:
                print(f"  {task}")
        return
    if selected in ("eval", "collect"):
        from .runtime import launch

        return launch(selected, argv[1:])
    sys.argv = ["tactalent " + selected, *argv[1:]]
    runpy.run_module(COMMANDS[selected][1], run_name="__main__")

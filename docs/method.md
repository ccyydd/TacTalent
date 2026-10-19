# RoboICL benchmark bindings

Pinned upstream: [Mosi-AI/RoboICL](https://github.com/Mosi-AI/RoboICL), commit `a7d652a9f8d4363aa90c0bda00004686af0589d8`. Vendored upstream files are preserved unchanged.

## Method and settings

The adapter retains the official demonstration selector, tool-call dialogue, action parsing and corrections, execution feedback, tracking guard and history management. It binds a single Panda arm using seven-number world-delta action rows and native IK / forced position control. There is no independent experience-summary or retrieval module.

- `vision_tactile` (default): task-authorized visual cameras and left/right tactile RGB.
- `vision`: the same visual cameras; tactile is removed from demonstrations and live observations.
- UniVTAC uses fixed + wrist cameras for Lift Can and Insert Tube, and fixed only for the other six tasks. OpenViTac uses fixed + wrist for all eleven tasks.
- Commands and demonstrations run at 60 Hz; each target is counted once and spans two 120 Hz physical substeps.
- One-shot uses 12 demonstration chunks and 12 live anchors. Covered few-shot tasks use 5 chunks per demonstration and 5 live anchors.
- Reasoning effort is `xhigh`. The inherited configured model is `gpt-6-astra`.

| Demonstration count | Tasks |
| --- | --- |
| 2 | Grasp Classify, Weight Classify, Hardness Classify, Roughness Classify, Roughness-Guided Regrasp, Empty-Can Selection |
| 3 | Insert Block v2: cube, half-cylinder, hexagon |
| 1 | All remaining tasks |

H is 5 by default, 10 for Lift Bottle, Put Bottle, Empty-Can Selection, Grasp Chip, Gear Assembly and both Insert Block tasks, and 4 for Grasp Classify. H does not change within an episode.

Task backgrounds and goals are provided by the benchmark profiles. Native success checks are retained except for the documented [Grasp Chip V5 protocol](roboicl_chip_evaluation_protocol.md). Evaluation budgets are explicitly recorded in `EVALUATION_COMMAND_LIMITS`; they do not change training limits.

## Preparation and evaluation

Use [the demonstration workflow](demonstrations.md) for dataset layout, source declarations, class-coverage ledgers and the unified `tactalent prepare` commands.

These commands require the environment described in the [root README](../README.md). Output must be inside `ASTRA_STORAGE_ROOT` (defaults to the working directory). Use fresh output directories.

After configuring the training sources described in the guide, build the reference bank:

```bash
export OUTPUT_ROOT="${ASTRA_STORAGE_ROOT:-$PWD}"
tactalent prepare build \
  --sources "$OUTPUT_ROOT/work/reference_sources.json" --icl-mode vision_tactile

bash scripts/evaluate.sh --benchmark OpenViTac --task turn_gear_pair \
  --icl-mode vision_tactile \
  --reference "$OUTPUT_ROOT/references/vision_tactile/OpenViTac/turn_gear_pair/default/reference.json" \
  --output "$OUTPUT_ROOT/eval_result/turn_gear_pair_vt" --seed 1000000
```

Use the same `--icl-mode` for preparation and execution. Demonstration action targets are approximated from adjacent measured states. Source, task, modality, calibration and protocol fingerprints are checked before evaluation.

## Evaluation plans

The full reviewed plan contains 2,100 episodes per modality: 800 UniVTAC and 1,300 OpenViTac. A 10% sample with sampling seed 42 selects 210 episodes: 80 + 130. Sampling preserves the configured class, shape and layout proportions.

```bash
tactalent suite --benchmark OpenViTac \
  --icl-mode vision_tactile --sample-ratio 0.1 --sample-seed 42 \
  --plan-only --output "$OUTPUT_ROOT/plans/openvitac_sample10_vt"
```

Execute a prepared catalog with `--catalog`, `--gpus` and an output directory. Reuse exact episode identities when comparing modalities. API interruptions support continuation while the original simulator process remains running.

The reference catalog defaults to `references/<icl-mode>/catalog.json` under `ASTRA_STORAGE_ROOT`. Override it with `--reference-catalog` or `ASTRA_REFERENCE_CATALOG`.

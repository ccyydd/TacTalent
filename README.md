<div align="center">
  <h1>TacTalent</h1>
  <h3>GPT-6 Astra Is a Native Visuo-Tactile Policy for Contact-Rich Manipulation</h3>
  <p>Chen Yang<sup>1,2</sup> · Linzhe Shi<sup>2</sup> · Hang Zhang<sup>2</sup> · Mu Xu<sup>2</sup> · Jiansheng Fan<sup>1</sup> · Chen Wang<sup>1,†</sup></p>
  <p><sup>1</sup>Tsinghua University &nbsp; <sup>2</sup>Amap, Alibaba Group<br><sup>†</sup>Corresponding author</p>
  <p>
    <a href="https://tactalent.github.io/?lang=en"><img src="assets/readme/links/project-page-en.svg?v=a3d66d4c0a" height="20" alt="Project Page"></a>
    <a href="https://github.com/ccyydd/SimpleTouch"><img src="assets/readme/links/code-simpletouch.svg?v=21db3f4a48" height="20" alt="Code (SimpleTouch)"></a>
    <a href="https://tactalent.github.io/records.html?lang=en"><img src="assets/readme/links/rollout-records-en.svg?v=63cda6f89a" height="20" alt="Rollout Records"></a>
    <a href="https://tactalent.github.io/gallery.html?lang=en&amp;benchmark=UniVTAC"><img src="assets/readme/links/univtac-gallery-en.svg?v=c2d6ff9a0d" height="20" alt="UniVTAC Video Gallery"></a>
    <a href="https://tactalent.github.io/gallery.html?lang=en&amp;benchmark=OpenViTac"><img src="assets/readme/links/openvitac-gallery-en.svg?v=ac7ca73da8" height="20" alt="OpenViTac Video Gallery"></a>
  </p>
</div>

**English** · [简体中文](README.zh-CN.md)

TacTalent shows that GPT-6 Astra can **directly understand and use visuotactile images for contact-rich manipulation**, without task-specific training. Across **8 UniVTAC tasks** and **11 OpenViTac tasks**, we investigate its ability to interpret tactile cues, adjust actions using contact feedback, and exploit subtle visual cues when touch is absent.

We use [RoboICL](https://github.com/Mosi-AI/RoboICL) as the few-shot closed-loop control framework. This repository provides benchmark integrations, demonstration preparation, evaluation scripts and rollout-review tools.

## 📊 Results

With vision and touch, Astra achieves **93.8%** mean success on UniVTAC and **81.8%** on OpenViTac. Vision-only performance reaches **76.3%** and **80.6%**, respectively. Both modes outperform the reported specialist baselines in the comparisons below.

![Benchmark results on UniVTAC and OpenViTac](assets/readme/benchmark-performance.png)

Mean task success (%). Astra uses a 10% stratified sample with sampling seed 42; baselines use their reported benchmark results. Full task results and analysis are on the [Project Page](https://tactalent.github.io/?lang=en).

## 🎬 Video Gallery

Browse demonstrations and evaluation videos in the [UniVTAC Gallery](https://tactalent.github.io/gallery.html?lang=en&benchmark=UniVTAC) and [OpenViTac Gallery](https://tactalent.github.io/gallery.html?lang=en&benchmark=OpenViTac), with synchronized action notes and commands.

## 🗂️ Repository Layout

The implementation targets UniVTAC and OpenViTac. They share the RoboICL controller while keeping task configuration and native interfaces separate.

| Directory | Contents |
| --- | --- |
| `src/tactalent/roboicl/` | Demonstration context, closed-loop control, evaluation and checkpoints |
| `src/tactalent/benchmarks/` | UniVTAC and OpenViTac profiles and native interfaces |
| `src/tactalent/robot/` | Panda state, kinematics and IK |
| `src/tactalent/_vendor/RoboICL/` | Pinned, unmodified upstream runtime |
| `configs/` | Local environment example and extra dependencies |
| `scripts/` | Evaluation and expert-collection entry points |
| `tests/` | Offline regression tests |
| `docs/` | Demonstration preparation, protocols and provenance |

## ⚙️ Environment Setup

Use Linux with an NVIDIA GPU, Conda, Git LFS, FFmpeg (`ffmpeg` and `ffprobe`) and build tools. The adapters target Python 3.10, Isaac Sim 4.5.0 and Isaac Lab 2.1.1. Install each benchmark in its own Conda environment. Start from the TacTalent repository root:

```bash
export TACTALENT_ROOT="$(pwd)"
export TACTALENT_BENCH_ROOT="$(realpath -m ../tactalent-benchmarks)"
mkdir -p "$TACTALENT_BENCH_ROOT"
git lfs install
```

### 1. Install the Benchmark

**UniVTAC**

```bash
git clone --branch main https://github.com/univtac/UniVTAC.git \
  "$TACTALENT_BENCH_ROOT/UniVTAC"
cd "$TACTALENT_BENCH_ROOT/UniVTAC"
conda activate base
bash scripts/install.sh
conda activate UniVTAC
cd "$TACTALENT_ROOT"
```

**OpenViTac**

```bash
git clone --branch main https://github.com/FVL-Repo/OpenViTac.git \
  "$TACTALENT_BENCH_ROOT/OpenViTac"
cd "$TACTALENT_BENCH_ROOT/OpenViTac"
conda activate base
bash scripts/install.sh
conda activate OpenViTac
cd "$TACTALENT_ROOT"
```

The installers set up simulation, cuRobo and the bundled TacEx/UIPC dependencies. Keep each benchmark’s bundled TacEx version. See the [UniVTAC installation guide](https://github.com/univtac/UniVTAC/blob/main/docs/Installation.md) and [OpenViTac installation guide](https://github.com/FVL-Repo/OpenViTac/blob/main/docs/Installation.md) for system prerequisites and manual setup.

### 2. Install Adapter Dependencies

Activate the benchmark environment and install the package and unified command from the TacTalent root:

```bash
python -m pip install -e .
```

For OpenViTac, install the additional adapter dependencies into the project runtime directory:

```bash
python -m pip install --no-deps --target .runtime/openvitac_python \
  -r configs/requirements-openvitac.txt
export ASTRA_EXTRA_PYTHONPATH="$TACTALENT_ROOT/.runtime/openvitac_python${ASTRA_EXTRA_PYTHONPATH:+:$ASTRA_EXTRA_PYTHONPATH}"
```

### 3. Configure Paths and API Access

Configure a local `.env` using [configs/env.example](configs/env.example). Point `ASTRA_UNIVTAC_ROOT` and `ASTRA_OPENVITAC_ROOT` to the installations above. Set `ASTRA_UNIVTAC_TRAIN_CAMERA_ROOT` to the camera source matching the demonstrations. Supply a complete HTTPS Responses endpoint and its credential through `ASTRA_API_ENDPOINT` and `ASTRA_API_KEY`.

```bash
cp configs/env.example .env
# Edit .env for your benchmark paths and API access.
set -a
source .env
set +a
export ASTRA_PYTHON="$(command -v python)"
export PYTHONPATH="$TACTALENT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OUTPUT_ROOT="${ASTRA_STORAGE_ROOT:-$TACTALENT_ROOT}"
```

After switching Conda environments, install TacTalent in that environment, reload the local configuration and update `ASTRA_PYTHON`.

## 🚀 Evaluation

Run from the repository root. List the supported tasks:

```bash
tactalent list
```

### Prepare Demonstrations

Prepare verified training recordings and coverage ledgers using the [demonstration guide](docs/demonstrations.md), then create the references and evaluation catalog with the unified entry point:

```bash
tactalent prepare training-config \
  --data-root "$OUTPUT_ROOT/work/datasets/UniVTAC"

tactalent prepare template \
  --output "$OUTPUT_ROOT/work/reference_sources.json"
# Configure the training ledgers described in docs/demonstrations.md.
tactalent prepare build \
  --sources "$OUTPUT_ROOT/work/reference_sources.json" --icl-mode vision_tactile
```

References are saved under `references/vision_tactile/` in `ASTRA_STORAGE_ROOT`. Use `--icl-mode vision` to generate a separate vision-only bank from the same source manifest. The suite automatically selects the catalog for the requested mode. For input checks, custom output locations and existing references, see the guide.

### Single Episode

```bash
tactalent eval --benchmark UniVTAC --task pull_out_key \
  --icl-mode vision_tactile \
  --reference "$OUTPUT_ROOT/references/vision_tactile/UniVTAC/pull_out_key/default/reference.json" --dry-run

GPU_ID=0 tactalent eval --benchmark UniVTAC --task pull_out_key \
  --icl-mode vision_tactile \
  --reference "$OUTPUT_ROOT/references/vision_tactile/UniVTAC/pull_out_key/default/reference.json" \
  --seed 1000000 --output "$OUTPUT_ROOT/eval_result/pull_out_key_vt"
```

### 10% Stratified Evaluation on Multiple GPUs

Save the evaluation plan before executing it. Reusing the sampling seed selects the same episodes; `--plan-only` does not start simulation or call the model.

```bash
tactalent suite --benchmark UniVTAC \
  --icl-mode vision_tactile --sample-ratio 0.1 --sample-seed 42 \
  --plan-only --output "$OUTPUT_ROOT/plans/univtac_sample10_vt"

tactalent suite \
  --catalog "$OUTPUT_ROOT/plans/univtac_sample10_vt/catalog.snapshot.json" \
  --gpus 0 1 --output "$OUTPUT_ROOT/eval_result/univtac_sample10_vt"
```

For OpenViTac, use its Conda environment, set `--benchmark OpenViTac` and supply the matching reference catalog. A 10% sample contains 80 UniVTAC episodes or 130 OpenViTac episodes. Omit `--sample-ratio` and `--sample-seed` for full evaluation.

The default input mode is `vision_tactile`. For vision-only evaluation, use `--icl-mode vision` for reference preparation and plan generation. Saved catalogs specify each episode’s mode. Outputs include results, videos, model actions and execution records in `--output`. See the [Grasp Chip protocol](docs/roboicl_chip_evaluation_protocol.md) for its scoring adaptation.

Shell entry points are available in [`scripts/evaluate.sh`](scripts/evaluate.sh) and [`scripts/collect.sh`](scripts/collect.sh), the latter for OpenViTac scripted experts.

## 🧪 Offline Checks

With the benchmark paths configured, run the regression suite without starting simulation or calling the model:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

## ⚖️ License

TacTalent’s original adapter and evaluation code is licensed under the [Apache License 2.0](LICENSE).

## 🤝 Acknowledgements

We thank the authors and maintainers of [RoboICL](https://github.com/Mosi-AI/RoboICL), [UniVTAC](https://github.com/univtac/UniVTAC) and [OpenViTac](https://github.com/FVL-Repo/OpenViTac).

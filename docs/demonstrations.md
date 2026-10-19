# Demonstrations and reference catalogs

Reference preparation runs offline from successful training recordings. The same source-selection manifest can generate vision-only and visuo-tactile references, preserving demonstration identities, per-task H and the upstream chunk selector.

## 1. Training recordings

Data collection and published datasets are described by the [UniVTAC data guide](https://github.com/univtac/UniVTAC#data-collection) and [OpenViTac collection guide](https://github.com/FVL-Repo/OpenViTac/blob/main/docs/Collection.md). Use the Isaac Sim 4.5 recordings for the benchmark installations in this repository.

Keep input recordings and ledgers under `ASTRA_STORAGE_ROOT`. The default storage root is the working directory. `work/` and generated `references/` directories are ignored by Git.

### UniVTAC source declaration

The audited training-source format is:

```text
work/datasets/UniVTAC/
  download_manifest.json
  pull_out_key/clean/0.hdf5
  pull_out_key/clean/1.hdf5
  ...
  grasp_classify/clean/0.hdf5
  ...
```

The download manifest records the upstream SHA-256, verified local SHA-256 and byte size of each source file:

```json
{
  "files": {
    "pull_out_key/clean/0.hdf5": {
      "upstream_sha256": "<SHA-256 from the trusted dataset source>",
      "verified_sha256": "<matching SHA-256 of the downloaded file>",
      "size": 123456
    }
  }
}
```

Keep the original recording bytes and their upstream hash records when organizing a dataset into this layout. Reference conversion checks these hashes and sizes. The training partition remains episode indices 0 through 49.

Create the source declaration:

```bash
tactalent prepare training-config \
  --data-root "$OUTPUT_ROOT/work/datasets/UniVTAC"
```

By default this writes `$OUTPUT_ROOT/work/training_sources/univtac.yaml`, which is also the default preparation input. For a different file, use `--output` and set `ASTRA_UNIVTAC_SOURCE_CONFIG` to the resulting YAML path. Individual preparation commands also accept `--source-config`.

Robot and asset paths derive from the benchmark roots:

| Configuration | Default |
| --- | --- |
| `ASTRA_UNIVTAC_ROOT` | UniVTAC installation containing the native task and collector |
| `ASTRA_UNIVTAC_URDF` | `$ASTRA_UNIVTAC_ROOT/assets/embodiments/franka/panda.urdf` |
| `ASTRA_UNIVTAC_TRAIN_CAMERA_ROOT` | Source matching the recorded cameras; defaults to `ASTRA_UNIVTAC_ROOT` |
| `ASTRA_UNIVTAC_TACEX_ASSETS` | `third_party/TacEx/source/tacex_assets/tacex_assets` under the training-camera source root |

### Successful training ledgers

OpenViTac uses explicit successful-training ledgers. The required top-level fields are `benchmark`, `task`, `variant`, `split`, `upstream_commit`, `split_basis`, `instruction` and `episodes`. Use the native recorded instruction and the benchmark revision represented by the recordings. The current OpenViTac revision is recorded in `UPSTREAM_COMMIT` in `src/tactalent/benchmarks/openvitac/profile.py`.

Each episode supplies its absolute HDF5 `path`, `sha256`, training `seed` and `success: true`. The input is checked against the recorded file hash. One-shot tasks use the first eligible ledger entry.

Few-shot tasks additionally require each entry's class or branch `coverage` and an auditable `coverage_basis`. Their configured coverage is:

| Benchmark / task | Coverage |
| --- | --- |
| UniVTAC / Grasp Classify | `rough`, `smooth` |
| OpenViTac / Weight Classify | `light`, `heavy` |
| OpenViTac / Hardness Classify | `soft`, `hard` |
| OpenViTac / Roughness Classify | `smooth`, `rough` |
| OpenViTac / Roughness-Guided Regrasp | `retain_initial`, `regrasp` |
| OpenViTac / Empty-Can Selection | `empty_found`, `filled_rejected` |
| OpenViTac / Insert Block v2 | `cube`, `half_cylinder`, `hexagon` |

UniVTAC few-shot entries identify their training `episode_id` in 0..49. OpenViTac entries use a training `seed` below 1,000,000. Each Insert Block v2 entry also specifies its `variant`. Coverage is selection metadata and is not inserted into model observations.

Example episode fields for an OpenViTac hardness ledger:

```json
{
  "seed": 0,
  "path": "/absolute/path/under/storage/hardness_classify/0.hdf5",
  "sha256": "<SHA-256 of the verified training file>",
  "success": true,
  "coverage": ["soft"],
  "coverage_basis": {
    "path": "/absolute/path/under/storage/training_result_0.json",
    "sha256": "<SHA-256 of the native training-condition evidence>"
  }
}
```

The corresponding ledger needs another distinct successful entry covering `hard`. The existing few-shot selector validates full coverage and training/evaluation separation.

## 2. Declare demonstration sources

Generate a source-selection manifest for all 19 tasks, or use `--benchmark UniVTAC` / `--benchmark OpenViTac` to select one suite:

```bash
tactalent prepare template \
  --output "$OUTPUT_ROOT/work/reference_sources.json"
```

UniVTAC one-shot entries explicitly select training episode 0. Edit the training-ledger paths to identify your verified few-shot and OpenViTac inputs. Manifest-level paths are relative to the manifest's directory; paths inside training ledgers are absolute.

For example, this manifest selects two tasks:

```json
{
  "schema_version": "tactalent_reference_sources_v1",
  "tasks": [
    {"benchmark": "UniVTAC", "task": "pull_out_key", "training_episode": 0},
    {"benchmark": "OpenViTac", "task": "hardness_classify",
     "training_ledger": "training_ledgers/OpenViTac/hardness_classify.json"}
  ]
}
```

A UniVTAC entry may specify `source_config` to override the default YAML. An Insert Block v2 entry without `variant` prepares all three variants from the same covered ledger; specifying `variant` prepares only that binding.

## 3. Prepare references and catalogs

Check the manifest, coverage and input paths before conversion:

```bash
tactalent prepare build \
  --sources "$OUTPUT_ROOT/work/reference_sources.json" \
  --icl-mode vision_tactile --check-only
```

Build either mode from that same manifest:

```bash
tactalent prepare build \
  --sources "$OUTPUT_ROOT/work/reference_sources.json" --icl-mode vision_tactile

tactalent prepare build \
  --sources "$OUTPUT_ROOT/work/reference_sources.json" --icl-mode vision
```

The default outputs are `references/vision_tactile/catalog.json` and `references/vision/catalog.json` under `ASTRA_STORAGE_ROOT`. Preparation routes each task through the existing one-shot/few-shot builder, validates every reference with the evaluation preflight, and records the source-manifest hash. A complete manifest produces 21 task/variant references per mode for the 19 logical tasks.

Use a fresh `--output` directory for an additional bank. For separate benchmark environments, prepare each benchmark with `--benchmark` and its own `--output`, then pass the generated catalog explicitly to evaluation.

To create a catalog from already prepared references:

```bash
tactalent prepare catalog \
  --references "$OUTPUT_ROOT/references/custom_bank" \
  --icl-mode vision_tactile \
  --output "$OUTPUT_ROOT/references/custom_bank/catalog.json"
```

Existing references pass the same task, modality, calibration, demonstration and source-fingerprint checks as evaluation.

## 4. Evaluate

The suite automatically selects the catalog for `--icl-mode`:

```bash
tactalent suite --benchmark UniVTAC \
  --icl-mode vision --sample-ratio 0.1 --sample-seed 42 \
  --plan-only --output "$OUTPUT_ROOT/plans/univtac_vision_sample10"
```

`--reference-catalog` overrides the default for one invocation. `ASTRA_REFERENCE_CATALOG` supplies a deployment-level override. Explicit saved evaluation catalogs passed through `--catalog` retain their stored episode modes and identities.

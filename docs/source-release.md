# Source and reproducibility

## Scope

The release supports **UniVTAC and OpenViTac**: 19 tasks with 21 task/shape
bindings. The implementation is an installable Python package under
`src/tactalent/`, exposed through the `tactalent` command. Benchmark profiles,
robot interfaces, reference preparation, evaluation and review tools have
separate modules. Environment examples and shell entry points live in
`configs/` and `scripts/`.

The source originates from experiment commit `cf60c50`. Deployment paths,
output locations and API access are configured through environment variables.
Source provenance is recorded in [`provenance/source.json`](provenance/source.json).

## Framework and references

The required RoboICL runtime is pinned to
`a7d652a9f8d4363aa90c0bda00004686af0589d8`. Its 28 retained files are unchanged
and checked against a SHA-256 manifest. The MIT license and upstream
attribution accompany those files.

The package reorganization preserves task text, camera choices, H, command
budgets, control cadence, demonstration coverage and the V5 chip scorer.
Source fingerprints follow the installed layout. Archived reference banks
remain immutable; use the [preparation workflow](demonstrations.md) to create
new validated banks after source changes.

## Validation

- 310 offline regression tests cover control accounting, native initialization,
  scoring, few-shot coverage, input modes, source preparation, checkpoints,
  provider errors, sampling, review generation and package entry points.
- All 21 task/shape bindings and their task context, cameras, H, demonstration
  contracts and calibration match the previous release.
- The two 10% plans with seed 42 select the same 210 episode identities per
  modality, including class, shape and layout conditions.
- A real UniVTAC training recording produces identical selected segments,
  action tensors and encoded model images in both input modes. Conversion and
  preflight complete without model calls.
- A wheel installs outside the source tree and exposes all command entry
  points, vendored runtime resources and the review template.

# TacTalent maintenance

The supported benchmarks are UniVTAC and OpenViTac. Benchmark task text, camera choices, demonstration coverage and control limits belong in their own profiles under src/tactalent/benchmarks/.

Keep the vendored RoboICL runtime byte-identical to its pinned source and retain its license and attribution. Changes to package layout must preserve scoring, H, cadence, model settings and selected demonstration content. Record and test intentional behavior changes separately.

Read deployment settings from the local environment. Keep credentials, private endpoints, server paths, experiment artifacts and raw request logs out of version control. Use offline tests for source-only changes.

Reference fingerprints must cover the installed source layout. Preserve archived references and results; rebuild and validate new references after source changes rather than disabling version checks.

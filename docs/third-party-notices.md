# Third-party attribution

## RoboICL — core policy framework

- Project: [RoboICL: Embodied In-Context Learning with GPT-6 Astra](https://github.com/Mosi-AI/RoboICL).
- Vendored version: [`a7d652a9f8d4363aa90c0bda00004686af0589d8`](https://github.com/Mosi-AI/RoboICL/tree/a7d652a9f8d4363aa90c0bda00004686af0589d8).
- Location: [`src/tactalent/_vendor/RoboICL/`](../src/tactalent/_vendor/RoboICL/).
- License: [MIT License](../src/tactalent/_vendor/RoboICL/LICENSE), copyright © 2026 RoboICL Authors.
- Provenance: [`UPSTREAM.json`](../src/tactalent/_vendor/RoboICL/UPSTREAM.json) and the retained-file hash manifest.

TacTalent uses RoboICL’s demonstration selection, dialogue-policy machinery,
history management, action parsing and correction, execution feedback and tracking
guards. The required runtime files are retained byte-for-byte with their copyright notice and license.
TacTalent’s additions are the UniVTAC / OpenViTac interfaces, vision-based tactile
observation extension, demonstration and evaluation preparation, and empirical study.

## Benchmarks and simulation dependencies

UniVTAC and OpenViTac supply native task and evaluation definitions. Their
repositories, datasets and assets are separate dependencies. Isaac Sim, Isaac Lab,
TacEx and cuRobo are also separate dependencies. Their licenses and access terms
remain those of their respective authors; this notice does not relicense them.


## TacTalent code

TacTalent's original adapter and evaluation code is licensed under the
[Apache License 2.0](../LICENSE). See [NOTICE](../NOTICE) for project attribution.

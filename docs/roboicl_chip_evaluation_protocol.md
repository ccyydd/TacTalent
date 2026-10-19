# Grasp Chip evaluation protocol

## Current completion rule

Grasp Chip evaluation uses `roboicl_chip_stable_placement_no_impact_v5`,
an adaptation of OpenViTac's `chip_physical_lifecycle_v3` lifecycle.
The native source is pinned to `a2d23ad8ae4351292bb6f51aad7dfdf481f074fe`.

Success requires a verified grasp and lift, followed by an intact chip stably
supported inside the target tray for **14 consecutive 120 Hz physical samples**
(approximately 0.117 seconds). The native footprint, support-contact, gap,
linear-speed and angular-speed checks determine stable support.
Gripper release and withdrawal are optional once placement is complete.

## Changes from the native lifecycle

| Condition | Native lifecycle | TacTalent evaluation |
| --- | --- | --- |
| Gripper orientation error above the native threshold | Terminal failure | Diagnostic event; grasp, lift and support evidence continue to accumulate |
| Slip, unstable grasp or off-plate event | Terminal failure | Recoverable event; a new attempt must establish its own grasp and lift evidence |
| Early release or slip over the target after verified grasp and lift | Terminal failure | Preserve the verified history and wait for stable placement |
| Plate overpress | Terminal failure | Record the measured overtravel; stable-support checks determine completion |
| Landing speed above the native impact threshold | Speed-triggered fracture and termination | Record the impact event and incoming speed; evaluate strain damage normally |
| Final release and withdrawal | Required | Optional after verified intact stable placement |
| Strain or squeeze fracture | Terminal failure | Terminal failure |
| Nonfinite or invalid physics | Invalid simulation | Invalid simulation |
| Exhausted command budget with an incomplete task | Failure | Failure |
| Unrecognized native failure reason | Terminal failure | Terminal failure |

All support evidence comes from physical samples. Repeated processing of the
same sample does not increase dwell. Loss of support or fracture can revoke
success if a diagnostic caller continues stepping after completion.
Outside-target recovery clears grasp/lift qualification and monitor references
before another attempt.

These rules apply to evaluation instances in both `vision` and `vision_tactile`
modes. Expert training collection keeps the native lifecycle. Other tasks retain
their native success checks.

## Execution and initialization

The evaluation budget is **600 commands at 60 Hz**, or 10 simulated seconds.
Each command spans two 120 Hz physical steps. The demonstration horizon is
H = 10; the 405-command training demonstration fits within this budget.
Training collection retains the native 3,600-command budget.

The unloaded GelSight reset repair checks sensor attachment before manipulation.
Its 1 mm attachment tolerance determines initialization validity. Material
parameters, friction, grip strength and strain-fracture thresholds retain their
native values.

## Result and reference records

`runtime.json`, `result.json` and native task metadata identify the protocol
through `evaluation_rules`, `success_contract` and physical trace snapshots.
`matches_upstream_protocol` is `false`, distinguishing this task's adapted
criterion from the native OpenViTac criterion. The legacy `native_success`
field contains the environment adapter's verdict; `evaluation_rules` identifies
the applicable protocol.

Reference preparation records the scorer's source fingerprint. A fresh
evaluation uses a matching validated reference while retaining the selected
training identities, H, segment starts, images and action tensors.
Archived reference banks and original evaluation records remain immutable.

### Saved-action reassessments

The reported chip results include reassessments of archived episodes under the
revised completion rule. Each derived record retains the original episode
identity, source verdict, command sequence, protocol, usage and replay evidence.
Saved-action replay executes previously generated commands; some V3 replays
also include explicitly recorded final-target holds. Replay adds no model
decisions and is associated with its original sample identity.

Original results and videos remain archived alongside the derived result and
updated replay video. Usage accounting includes the source attempts.
The final chip summaries contain 10/10 successes for each input mode under the
adapted protocol.

## Revision and validation history

| Version | Change |
| --- | --- |
| V1 | Recover selected manipulation events and disable the plate-overpress gate |
| V2 | Record speed-only impact events without applying impact fracture |
| V3 | Complete on verified intact stable placement without requiring release or withdrawal |
| V4 | Make orientation errors diagnostic and set the 600-command evaluation budget |
| V5 | Preserve verified grasp/lift history after an in-target slip |

Offline regressions cover lifecycle transitions, impact handling, initialization,
terminal errors, command accounting and task-context consistency. Reference
checks compare demonstration identities, observations, actions and selection.

Same-seed native-monitor replays confirmed V4 placement for seeds 10001 and 10067
at saved commands 176 and 182. The V5 correction confirmed vision seed 10095 at
saved command 184 / physical step 425, following 14 stable-support samples.
Its initialization matched all six recorded scene parameters, its command
prefix matched exactly, and sampled physical-state error was below
6.54e-13 m. These V4/V5 replays used only the original command prefixes.

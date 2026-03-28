# NOTES v9 — stratified off-policy SAC + grid evaluation

This revision focuses on the two issues that became clear in v8:

1. the off-policy branch needed a **better replay mix** so that medium/hazard states were not drowned out by easy episodes,
2. the project needed a **structured evaluation surface** rather than one aggregate success number.

## What changed

### 1) Stratified replay for the SAC branch
A new replay module was added:

- `shared_control_rl/replay.py`

Key behavior:

- stores `demo`, `hazard`, and `difficulty` tags for every transition,
- samples batches with configurable
  - `demo_sample_frac`
  - `hazard_sample_frac`,
- assigns a simple hazard severity score from `h`, `ttc`, collision flag, and takeover level,
- slightly up-weights TD loss on hazard transitions.

This is still a minimal replay design, not full PER. The goal is to keep the implementation simple and interpretable while improving sample quality for the authority-allocation problem.

### 2) Better SAC training loop
Updated file:

- `scripts/train_sac_torch.py`

New ideas added:

- **heuristic demo seeding**,
- **driver-only failure seeding**,
- **linear BC-coefficient decay**,
- **guided exploration** using the heuristic allocator early in training,
- replay statistics logged into training history,
- richer metadata saved with the checkpoint.

### 3) Grid evaluation over scenario families
New file:

- `scripts/evaluate_policy_grid.py`

This evaluates a policy over a fixed matrix:

- difficulty: `easy / medium / hard`
- driver profile: `late_linear / late_aggressive / frozen / wrong_initial`

Outputs:

- per-episode CSV,
- grouped summary CSV,
- success / collision / min-h heatmaps.

### 4) Forced scenario rendering
Rendering/evaluation scripts now accept fixed `difficulty` and `driver_profile`, which makes it much easier to compare policies on the exact same case.

## Smoke-training artifact included in v9
Checkpoint:

- `artifacts/v9_sac_best.pt`

This checkpoint was obtained from a short stratified SAC run and selected using the built-in eval score. The checkpoint metadata reports:

- best checkpoint at step **400**,
- eval success **1.00**,
- eval collision **0.00**,
- eval min_h **0.0885**,
- eval mean takeover **0.7067**.

Interpretation: the short-run SAC can already find a collision-free solution on the internal periodic eval set, but it is still too takeover-heavy.

## Fixed-grid comparison (12 cells, 1 rollout per cell)
Overall comparison from `artifacts/v9_policy_overall_comparison.csv`:

| policy | success | collision | min_h | mean takeover |
|---|---:|---:|---:|---:|
| driver | 0.0833 | 0.8333 | 0.0227 | 0.0000 |
| heuristic | 0.7500 | 0.2500 | 0.0832 | 0.4955 |
| SAC v9 | 0.7500 | 0.2500 | -0.0249 | 0.6168 |

### What this means

- The **driver-only** policy still fails in most cells, which is expected and desirable for the shared-control problem setup.
- The **SAC v9 smoke model matches the heuristic baseline on overall success/collision** on this grid.
- The current SAC model is still **more aggressive** than the heuristic, with noticeably higher takeover.
- The remaining weakness is concentrated in the **medium-difficulty cells**, not the easy or hard extremes.

That last point is important. The new grid evaluation exposed that the open problem is no longer “can it solve hard hazard cases at all?” but rather “can it keep minimal intervention in the ambiguous medium regime?”

## Representative rendered case
Scenario:

- difficulty = `hard`
- driver profile = `frozen`
- seed = `262`

Artifacts:

- `artifacts/animations_v9/hard_frozen_driver.gif`
- `artifacts/animations_v9/hard_frozen_heuristic.gif`
- `artifacts/animations_v9/hard_frozen_sac.gif`

Summary:

- driver: collision
- heuristic: success, min_h ≈ 0.0034, mean takeover ≈ 0.5038
- SAC v9: success, min_h ≈ 0.0038, mean takeover ≈ 0.6743

So the SAC controller is capable of solving the canonical “frozen drowsy driver” case, but it is still pulling authority earlier/harder than desired.

## Most useful next step after v9
The next bottleneck is no longer basic capability. It is **minimal-intervention shaping**.

The most promising v10 directions are:

1. add **takeover-aware imitation regularization** only in medium-risk states,
2. switch from uniform guided exploration to **state-conditional guided exploration**,
3. train/evaluate with **more than one rollout per grid cell** to reduce seed sensitivity,
4. optionally add a small **sequence encoder** for the SAC actor/critic instead of pure stacked observations.

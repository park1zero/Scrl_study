# NOTES_v8

## What changed from v7

### 1. Curriculum mixture

The environment now supports a progress-dependent mixture over:
- `easy`
- `medium`
- `hard`

As progress increases, hard episodes and late-start hazard-focused episodes become more common.

### 2. Driver population

The old single sleepy-driver family has been expanded into four profiles:
- `late_linear`
- `late_aggressive`
- `frozen`
- `wrong_initial`

The last profile supports a short wrong-way initial steering response.

### 3. Off-policy SAC branch

Added `scripts/train_sac_torch.py` and `scripts/evaluate_sac_torch.py`.

The SAC branch includes:
- twin critics
- entropy-temperature tuning
- demonstration-seeded replay buffer
- optional behavior-cloning regularization on demo samples
- optional PPO checkpoint initialization for the SAC actor

## Included smoke artifacts

- `artifacts/v8_curriculum_population_samples.csv`
- `artifacts/v8_curriculum_population_counts.png`
- `artifacts/v8_driver_curriculum_sweep.csv`
- `artifacts/v8_heuristic_curriculum_sweep.csv`
- `artifacts/v8_sac_curriculum_sweep.csv`
- `artifacts/v8_policy_comparison_summary.csv`
- `artifacts/v8_policy_comparison_summary.png`
- `artifacts/v8_sac_best.pt`
- `artifacts/animations_v8/index.html`

## Interpretation of the smoke results

The smoke SAC checkpoint is not yet stronger than the heuristic baseline in aggregate curriculum sweeps. That is expected at this stage because:
- training was intentionally short
- the new driver population is materially harder than the old single-driver setup
- late-start curriculum episodes create many near-terminal decisions

The important part is that the off-policy path now exists, runs end-to-end, and can already solve some hard sampled episodes that the heuristic misses.

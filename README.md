# SBW shared-control RL starter — v11

This package extends the v10 sequence-aware off-policy branch with a **conservative fine-tuning** path:

- initialize the GRU SAC actor from `artifacts/v10_pretrain_only.pt`
- delay actor updates so the critic can warm up first
- freeze an anchor copy of the pretrained actor and regularize against it during RL updates
- keep medium-risk imitation retention stronger than hard-risk retention

## Main v11 additions

1. **Conservative SAC initialization**
   - `--init-actor-from-sac`
   - `--init-critic-from-sac`
   - `--init-alpha-from-sac`
   - `--actor-learning-starts`

2. **Frozen anchor regularization**
   - `--anchor-after-pretrain`
   - `--anchor-coef-start` / `--anchor-coef-end`
   - `--medium-anchor-boost` / `--hard-anchor-scale`

3. **Included v11 result set**
   - `artifacts/v11_conservative_best.pt`
   - `artifacts/v11_policy_overall_comparison.csv`
   - `artifacts/v11_policy_overall_comparison.png`
   - `artifacts/animations_v11/index.html`

## Quick start: v11 conservative fine-tuning

```bash
python scripts/train_sac_torch.py   --total-steps 80   --encoder-type gru   --history-stack 4   --curriculum   --driver-population   --seed-demo-episodes 8   --seed-driver-episodes 6   --pretrain-bc-steps 0   --init-actor-from-sac artifacts/v10_pretrain_only.pt   --anchor-after-pretrain   --actor-lr 8e-5   --critic-lr 3e-4   --learning-starts 40   --actor-learning-starts 60   --start-random-steps 0   --guided-explore-prob-start 0.16   --guided-explore-prob-end 0.05   --demo-sample-frac 0.30   --hazard-sample-frac 0.40   --bc-coef-start 0.26   --bc-coef-end 0.16   --anchor-coef-start 0.30   --anchor-coef-end 0.20   --medium-bc-boost 1.0   --hard-bc-scale 0.40   --medium-anchor-boost 0.90   --hard-anchor-scale 0.25   --best-out artifacts/v11_conservative_best.pt   --out artifacts/v11_conservative_last.pt
```

## Quick read of the included v11 result

On the included 12-cell grid, **v11 conservative SAC** matches the heuristic's success / collision rate while reducing mean takeover from **0.4955** to **0.4565** and increasing mean `min_h` from **0.0832** to **0.1037**.

---

# SBW shared-control RL starter — v10

This package implements a lightweight indirect shared-control testbed for steering assistance with:

- constant-speed dynamic bicycle + separate SBW `delta_swa` / `delta_rwa`
- split-delay drowsy-driver model
- driver population randomization
  - `late_linear`
  - `late_aggressive`
  - `frozen`
  - `wrong_initial`
- shooting-based obstacle-avoidance MPC surrogate
- one-step barrier-based lambda safety filter
- history-stacked observations for partial observability
- **on-policy PPO** path (`scripts/train_ppo_torch.py`)
- **off-policy SAC** path (`scripts/train_sac_torch.py`)
- curriculum sampler over `easy / medium / hard` episodes with optional late-start hazard focusing
- **sequence-aware SAC encoder** (`encoder_type=gru`)
- **risk-shaped imitation regularization** for medium-risk demo transitions
- **actor BC pretraining** before online SAC fine-tuning

## Main v10 additions

1. **Sequence-aware off-policy branch**
   - SAC actor / critic can use a GRU encoder over the stacked observation history
   - keeps the existing flattened-history environment interface

2. **Medium-risk imitation shaping**
   - demo BC weights are stronger in medium-risk cells
   - hard-risk demo states are down-weighted to allow deviation when stronger takeover is needed

3. **Sequence actor pretraining**
   - the off-policy branch can pretrain directly on heuristic replay data before RL updates
   - this makes short smoke runs much less brittle than random-start sequence SAC

## Quick start

### Environment smoke test

```bash
python scripts/smoke_test_env.py
```

### Sequence-aware SAC pretrain only

```bash
python scripts/train_sac_torch.py \
  --total-steps 0 \
  --encoder-type gru \
  --history-stack 4 \
  --curriculum \
  --driver-population \
  --pretrain-bc-steps 300 \
  --out artifacts/v10_pretrain_only.pt
```

### Sequence-aware SAC fine-tune

```bash
python scripts/train_sac_torch.py \
  --total-steps 120 \
  --encoder-type gru \
  --history-stack 4 \
  --curriculum \
  --driver-population \
  --pretrain-bc-steps 250 \
  --best-out artifacts/v10_sac_best.pt \
  --out artifacts/v10_sac_last.pt
```

### Grid evaluation

```bash
python scripts/evaluate_policy_grid.py \
  --policy sac \
  --model artifacts/v10_pretrain_only.pt \
  --history-stack 4 \
  --episodes-per-cell 1
```

### Render a hard/frozen scenario

```bash
python scripts/render_animation.py \
  --policy sac \
  --model artifacts/v10_pretrain_only.pt \
  --history-stack 4 \
  --difficulty hard \
  --driver-profile frozen \
  --curriculum-progress 1.0 \
  --out artifacts/animations_v10/hard_frozen_pretrain.gif
```

## Included smoke artifacts

- `artifacts/v10_policy_overall_comparison.csv`
- `artifacts/v10_policy_overall_comparison.png`
- `artifacts/v10_smoke_summary.json`
- `artifacts/v10_sac_best_metadata.json`
- `artifacts/v10_pretrain_only_metadata.json`
- `artifacts/animations_v10/index.html`

## Notes

- The obstacle-avoidance block is still a dependency-light shooting MPC surrogate, not a full CasADi nonlinear MPC.
- The lambda safety filter is still a discrete grid projection, not an analytic CBF-QP.
- In the included smoke results, **GRU + BC pretrain** matches the heuristic fairly closely on the 12-cell grid.
- The short **SAC fine-tune** lowers intervention in some cases but is not yet more robust than the heuristic.

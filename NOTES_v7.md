# v7 notes — history-stacked policies on the split-delay drowsy driver

## What changed

This revision adds a lightweight **observation history stack** on top of the v6 split-delay driver model.
The main motivation is partial observability: the authority allocator only sees the current state,
while the driver behaviour depends on delayed perception and delayed motor release.
Stacking recent observations makes it easier for a feed-forward policy to infer whether the driver is
about to react late and aggressively.

### New code paths

- `shared_control_rl/envs/history_stack.py`
- `shared_control_rl/envs/factory.py`
- checkpoint metadata now stores `history_stack`
- scripts updated to accept `--history-stack` and infer it from checkpoints when possible
- `scripts/run_history_stack_pipeline.py` added for BC -> PPO -> eval -> render orchestration

## Key implementation notes

1. **History stack wrapper**
   - Repeats the reset observation to fill the stack.
   - Returns `[o_{t-k+1}, ..., o_t]` as one flattened vector.
   - Works with both gymnasium and the local fallback Box implementation.

2. **Checkpoint metadata**
   - BC / PPO checkpoints now save `history_stack` in `metadata`.
   - Evaluation, sweep, and render scripts can infer the correct stacked observation size.

3. **Best-checkpoint saving for PPO**
   - Short PPO runs were unstable and could quickly move away from the BC warm-start.
   - `train_ppo_torch.py` now supports `--best-out` and `--best-metrics-json`.
   - The selection score prioritizes success, then low collision, then larger barrier margin,
     then smaller takeover, then return.

## Results in this package

Randomized 8-episode sweeps:

- driver only:
  - success = 0.125
  - collision = 0.875
  - min_h = -0.1305
- heuristic authority:
  - success = 1.000
  - collision = 0.000
  - min_h = 0.2768
  - mean takeover = 0.4827
- history-stack BC warm-start (`history_stack=4`):
  - success = 1.000
  - collision = 0.000
  - min_h = 0.2007
  - mean takeover = 0.4615
- history-stack PPO best checkpoint (`history_stack=4`):
  - success = 1.000
  - collision = 0.000
  - min_h = 0.2105
  - mean takeover = 0.4511

## Interpretation

The **history-stacked BC** policy is already strong and consistent.
A short PPO fine-tune does not yet beat the heuristic on barrier margin, but the saved **best PPO checkpoint**
keeps the randomized success rate at 1.0 and slightly reduces mean takeover compared with the BC warm-start.

The main takeaway from v7 is not that PPO is fully solved — it is that the project now has the right scaffolding
for partially observed shared-control learning:

- split-delay driver model
- history-stacked observations
- checkpoint-aware training/evaluation scripts
- best-model capture during unstable PPO fine-tuning

## Recommended next step

Use the v7 history-stack scaffolding and replace the current short PPO smoke run with either:

1. a longer PPO schedule with periodic evaluation and early stopping, or
2. a KL-constrained / imitation-regularized fine-tuning stage so the actor does not drift away from the BC prior too fast.

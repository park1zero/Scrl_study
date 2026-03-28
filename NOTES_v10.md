# NOTES_v10

## Main changes from v9

### 1. Sequence-aware SAC backbone
- `shared_control_rl/torch_sac.py`
- SAC actor / critic now support `encoder_type=gru`
- the flattened history stack is reshaped to `[B, T, obs_dim_base]`
- a compact GRU encodes short-term temporal context before the policy / Q heads

This keeps the existing history-stack environment interface but gives the off-policy branch a temporal encoder instead of a pure feedforward MLP.

### 2. Medium-risk imitation shaping
- `shared_control_rl/replay.py`
- `scripts/train_sac_torch.py`

Replay batches now include `hazard_severity`.
The actor BC regularizer is weighted by risk regime:
- medium-risk demo transitions get **higher** imitation weight
- hard-risk demo transitions get **down-weighted** imitation pressure

The intent is:
- keep the policy closer to the heuristic in the ambiguous medium regime
- allow more deviation in hard regimes where stronger takeover may be justified

### 3. Sequence-actor pretraining
- `scripts/train_sac_torch.py`

Added `--pretrain-bc-steps` / `--pretrain-batch-size`.
Before online SAC updates, the GRU actor can be pretrained directly on heuristic replay data.
This makes short smoke runs much more meaningful than starting sequence SAC completely from scratch.

## Smoke results included in this package

### Grid evaluation: 12-cell difficulty × driver-profile grid
- **driver only**
  - success: **0.0833**
  - collision: **0.8333**
- **heuristic**
  - success: **0.7500**
  - collision: **0.2500**
  - mean takeover: **0.4955**
- **sequence pretrain only**
  - success: **0.7500**
  - collision: **0.2500**
  - mean takeover: **0.5083**
- **sequence SAC fine-tune**
  - success: **0.4167**
  - collision: **0.5833**
  - mean takeover: **0.3183**

Interpretation:
- the **GRU + BC pretrain** branch already reproduces the heuristic fairly closely
- the short **SAC fine-tune** reduces takeover on the grid, but loses too much robustness
- so at this stage, the v10 off-policy branch is most useful as a **sequence-aware imitation + cautious RL fine-tuning** path, not yet a clear replacement for the heuristic

### Best short-run SAC checkpoint metadata
`artifacts/v10_sac_best_metadata.json`

Best evaluation stored during short training:
- best step: **50**
- eval success: **1.00**
- eval collision: **0.00**
- eval min_h: **0.2275**
- eval mean takeover: **0.4655**

That checkpoint looks good on its internal eval slice, but the full grid evaluation shows weaker generalization than the heuristic / pretrain-only actor.

## Files to look at first
- `shared_control_rl/torch_sac.py`
- `shared_control_rl/replay.py`
- `scripts/train_sac_torch.py`
- `artifacts/v10_smoke_summary.json`
- `artifacts/v10_policy_overall_comparison.csv`
- `artifacts/animations_v10/index.html`

## Suggested next move after v10
A sensible v11 direction would be:
1. keep GRU actor / critic
2. keep BC pretrain
3. fine-tune only from the pretrained checkpoint
4. lower RL step size and/or increase imitation retention in medium-risk cells
5. evaluate with larger per-cell sample counts before claiming improvement over the heuristic

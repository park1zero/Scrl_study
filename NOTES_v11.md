# NOTES_v11

## Main changes from v10

### 1. Conservative fine-tuning from the sequence-pretrained actor
- `scripts/train_sac_torch.py`
- Added `--init-actor-from-sac` so the off-policy branch can start from `artifacts/v10_pretrain_only.pt`
- Added separate `--actor-lr` / `--critic-lr`
- Added `--actor-learning-starts` so the critic can warm up before the actor moves

### 2. Frozen anchor regularization
- `scripts/train_sac_torch.py`
- Added `--anchor-after-pretrain`
- The trainer now freezes a copy of the initialized / pretrained actor and adds an anchor loss during SAC updates
- The anchor is stronger in medium-risk states and relaxed in hard-risk states

### 3. v11 smoke result
Using the included conservative settings:
- initialized from `artifacts/v10_pretrain_only.pt`
- `total_steps=80`
- GRU encoder, history stack 4
- curriculum + driver population enabled
- delayed actor updates and stronger BC / anchor retention

### 12-cell difficulty × driver-profile grid (1 episode per cell)
- **driver only**
  - success: **0.0833**
  - collision: **0.8333**
- **heuristic**
  - success: **0.7500**
  - collision: **0.2500**
  - mean takeover: **0.4955**
  - mean min h: **0.0832**
- **sequence pretrain only**
  - success: **0.7500**
  - collision: **0.2500**
  - mean takeover: **0.5083**
  - mean min h: **0.0957**
- **v11 conservative SAC**
  - success: **0.7500**
  - collision: **0.2500**
  - mean takeover: **0.4565**
  - mean min h: **0.1037**

Interpretation:
- v11 preserves the same overall success / collision rate as the heuristic and the pretrained sequence actor on the 12-cell grid
- compared with the heuristic, v11 reduces mean takeover by about **0.039** while also slightly increasing mean `min_h`
- compared with the pretrain-only actor, v11 reduces mean takeover by about **0.052**

## Files to inspect first
- `scripts/train_sac_torch.py`
- `artifacts/v11_policy_overall_comparison.csv`
- `artifacts/v11_policy_overall_comparison.png`
- `artifacts/v11_conservative_best_metadata.json`
- `artifacts/v11_smoke_summary.json`
- `artifacts/animations_v11/index.html`

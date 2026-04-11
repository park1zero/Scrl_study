# Final shared-control authority-allocation package with JBK-style driver

This package is the **final integrated RL authority-allocation branch** from the
conversation, repackaged without the step-by-step study scaffolding.

## What is included

- **SBW lateral plant** on the road-wheel / steering-wheel split channels
- **JBK-style preview driver** with:
  - near/far two-point preview structure
  - compensation / prediction terms
  - neuromuscular lag
  - drowsiness overlay (perception delay, motor delay, attention loss)
  - late hazard burst steering
- **Shared-control authority allocation**
  - driver RWA-equivalent and automation command are blended as
    `delta_cmd = lambda * delta_drv_rwa + (1-lambda) * delta_auto_rwa`
  - RL action is the authority-rate command
- **Teacher heuristic -> BC warm-start -> off-policy SAC fine-tune** pipeline
- **Best checkpoint** artifacts from the JBK-integrated run

## Current algorithm in this package

Observation uses a 4-step history stack:

z_k = [o_{k-3}, o_{k-2}, o_{k-1}, o_k]

Policy outputs a continuous action:

a_k ~ pi_theta(. | z_k)

Lambda update:

lambda_{k+1} = clip(lambda_k + lambda_rate_max * dt * a_k, 0, 1)

Shared steering command:

delta_cmd,k = lambda_{k+1} * delta_drv_rwa,k + (1-lambda_{k+1}) * delta_auto_rwa,k

## Important note

This package follows the **implemented final RL authority-allocation branch** from
the conversation. It does **not** add the later MPC+barrier replacement of the
automation controller. The automation block here is the lightweight obstacle-avoidance
steering law used in the trained JBK-integrated experiments.

## Key files

- `driver_jbk.py`: structured JBK-style preview driver with drowsiness overlay
- `env.py`: shared-control RL environment
- `collect_teacher_data.py`: collect teacher actions
- `train_bc.py`: BC warm-start
- `train_sac.py`: off-policy SAC fine-tune with BC/anchor regularization
- `evaluate_sac.py`: compare driver / heuristic / BC / SAC
- `run_final_pipeline.py`: end-to-end run script

## Artifacts included

- `artifacts/jbk_teacher_dataset.npz`
- `artifacts/jbk_bc_actor.pt`
- `artifacts/jbk_sac_best.pt`
- `artifacts/jbk_sac_last.pt`
- training/evaluation plots and summaries

## Quick start

### Use the included best checkpoint

```bash
python evaluate_sac.py --bc-ckpt artifacts/jbk_bc_actor.pt --sac-ckpt artifacts/jbk_sac_best.pt --driver-model jbk --episodes 24
```

### Re-run the whole integrated pipeline

```bash
python run_final_pipeline.py
```

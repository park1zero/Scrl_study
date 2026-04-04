# RL Study Step 3 — Fixed-\(\lambda\) Shared Control Baselines

This step is still **before RL**.
The goal is to understand the shared-control structure itself:

\[
\delta_{\mathrm{cmd}} = \lambda\,\delta_{\mathrm{drv}}^{\mathrm{rwa}} + (1-\lambda)\,\delta_{\mathrm{auto}}^{\mathrm{rwa}}
\]

where

- \(\delta_{\mathrm{drv}}^{\mathrm{rwa}}\): driver-equivalent road-wheel steering
- \(\delta_{\mathrm{auto}}^{\mathrm{rwa}}\): automation steering
- \(\delta_{\mathrm{cmd}}\): final road-wheel steering sent to the chassis
- \(\lambda\in[0,1]\): driver authority

## Files

- `vehicle.py` — minimal SBW lateral vehicle model
- `driver.py` — split-delay drowsy-driver model from Step 2
- `automation.py` — lightweight automation steering law (not MPC yet)
- `geometry.py` — obstacle barrier function
- `simulate_shared_baselines.py` — compares driver / automation / fixed-shared baselines
- `sweep_fixed_lambda.py` — sweeps constant \(\lambda\) values to see what fixed blending does

## Automation controller used in this step

The automation is intentionally simple so that the structure is easy to study.
It uses

1. a temporary lateral offset reference while approaching the obstacle
2. a small repulsive steering term when the obstacle gets close

The control law is

\[
\delta_{\mathrm{auto}}
=
K_y (y_{\mathrm{ref}} - y)
+
K_\psi(-\psi)
-
K_{vy} v_y
-
K_r r
+
u_{\mathrm{rep}}
\]

This is **not** the final MPC yet. It is just a clean baseline that makes the
shared-control blending behaviour visible.

## Run

```bash
cd /mnt/data/rl_study_step3_shared_control
python simulate_shared_baselines.py
python sweep_fixed_lambda.py
```

## What to check

### Baseline comparison

You should see:
- `driver` baseline fails (late drowsy response)
- `automation` baseline succeeds
- `shared` with a fixed \(\lambda\) can also succeed

### Lambda sweep

The sweep is important because it shows that **fixed authority allocation is not
always robust**. Some \(\lambda\) values work, some fail, and the relation need not
be monotonic.

That is exactly the motivation for moving later to
- dynamic authority allocation
- safety filtering
- RL-based scheduling of authority

## Why this step matters before RL

Once RL enters the project, the policy will *not* output steering directly.
Instead it will choose how \(\lambda\) changes over time.

So before touching RL, make sure you are comfortable with these ideas:

1. how driver steering is converted from sfa to equivalent RWA
2. how automation steering is generated
3. how blending changes the final road-wheel command
4. why a fixed \(\lambda\) is often too limited

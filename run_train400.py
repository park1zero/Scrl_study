import argparse, json
from pathlib import Path
from scripts.train_sac_torch import SACTrainer, plot_training
from shared_control_rl.torch_sac import save_checkpoint

args = argparse.Namespace(
    total_steps=400, seed=42, device='cpu', torch_threads=1, hidden_dim=128,
    history_stack=4, randomize=False, curriculum=True, driver_population=True,
    buffer_size=60000, batch_size=128, start_random_steps=150, learning_starts=250,
    updates_per_step=1, seed_demo_episodes=14, seed_driver_episodes=10,
    demo_sample_frac=0.20, hazard_sample_frac=0.35, hazard_threshold=0.55,
    guided_explore_prob_start=0.30, guided_explore_prob_end=0.05, lr=3e-4,
    alpha_lr=3e-4, init_alpha=0.18, gamma=0.99, tau=0.01,
    bc_coef_start=0.16, bc_coef_end=0.03, hazard_td_weight=0.35,
    max_grad_norm=1.0, eval_every=200, eval_episodes=4,
    out='artifacts/debug_last.pt', best_out='artifacts/debug_best.pt', curve_out='artifacts/debug_curve.png',
    metrics_json='artifacts/debug_metrics.json', init_ppo=None,
)
trainer = SACTrainer(args)
history, metrics = trainer.train()
print('TRAIN_DONE')
print(json.dumps(metrics, indent=2))

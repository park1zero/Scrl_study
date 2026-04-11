from __future__ import annotations

from pathlib import Path

from collect_teacher_data import collect_dataset
from train_bc import train as train_bc
from evaluate_bc import evaluate as eval_bc
from train_sac import train as train_sac
from evaluate_sac import evaluate as eval_sac


def main() -> None:
    root = Path('artifacts')
    dataset = root / 'jbk_teacher_dataset.npz'
    bc_ckpt = root / 'jbk_bc_actor.pt'
    bc_curve = root / 'jbk_bc_training_curve.png'
    bc_summary = root / 'final_bc_policy_summary.csv'
    bc_preview = root / 'final_bc_policy_preview.png'
    bc_compare = root / 'final_bc_rollout_compare.png'
    sac_best = root / 'jbk_sac_best.pt'
    sac_last = root / 'jbk_sac_last.pt'
    sac_curve = root / 'final_sac_training_curve.png'
    sac_metrics = root / 'final_sac_metrics.json'
    sac_summary = root / 'final_policy_summary.csv'
    sac_preview = root / 'final_policy_preview.png'
    sac_compare = root / 'final_rollout_compare.png'
    sac_animation = root / 'final_policy_compare_animation.gif'

    collect_dataset(episodes=180, output_path=dataset, driver_model='jbk', seed0=100)
    train_bc(dataset_path=dataset, ckpt_path=bc_ckpt, curve_path=bc_curve, epochs=18, batch_size=256, lr=1e-3, seed=0)
    eval_bc(ckpt=bc_ckpt, driver_model='jbk', episodes=24, summary_csv=bc_summary, preview_png=bc_preview, compare_png=bc_compare)
    train_sac(bc_ckpt=bc_ckpt, best_out=sac_best, last_out=sac_last, curve_png=sac_curve, metrics_json=sac_metrics, driver_model='jbk', total_steps=2200, demo_episodes=24, batch_size=128, seed=0)
    eval_sac(bc_ckpt=bc_ckpt, sac_ckpt=sac_best, driver_model='jbk', episodes=24, summary_csv=sac_summary, preview_png=sac_preview, compare_png=sac_compare, animation_path=sac_animation)


if __name__ == '__main__':
    main()

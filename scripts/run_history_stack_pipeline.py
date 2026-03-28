#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import subprocess


def run(cmd: list[str], cwd: Path) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end history-stack BC + PPO + evaluation pipeline.")
    parser.add_argument("--history-stack", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bc-episodes", type=int, default=16)
    parser.add_argument("--bc-epochs", type=int, default=10)
    parser.add_argument("--ppo-steps", type=int, default=12_000)
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--tag", type=str, default="history_stack")
    args = parser.parse_args()

    root = ROOT
    art = root / "artifacts"
    anim = art / f"animations_{args.tag}"
    art.mkdir(exist_ok=True)
    anim.mkdir(exist_ok=True)

    bc_ckpt = art / f"{args.tag}_bc_actorcritic.pt"
    ppo_ckpt = art / f"{args.tag}_ppo_actorcritic.pt"

    cmd = [sys.executable, "scripts/train_bc_warmstart.py", "--history-stack", str(args.history_stack), "--seed", str(args.seed), "--episodes", str(args.bc_episodes), "--epochs", str(args.bc_epochs), "--out", str(bc_ckpt), "--dataset-out", str(art / f"{args.tag}_bc_dataset.npz"), "--curve", str(art / f"{args.tag}_bc_curve.png"), "--meta", str(art / f"{args.tag}_bc_meta.json")]
    if args.randomize:
        cmd.append("--randomize")
    run(cmd, cwd=root)

    cmd = [sys.executable, "scripts/train_ppo_torch.py", "--history-stack", str(args.history_stack), "--seed", str(args.seed), "--total-steps", str(args.ppo_steps), "--init-model", str(bc_ckpt), "--out", str(ppo_ckpt), "--curve", str(art / f"{args.tag}_ppo_curve.png"), "--metrics-json", str(art / f"{args.tag}_ppo_metrics.json")]
    if args.randomize:
        cmd.append("--randomize")
    run(cmd, cwd=root)

    cmd = [sys.executable, "scripts/evaluate_torch_policy.py", "--model", str(ppo_ckpt), "--seed", str(args.seed), "--out", str(art / f"eval_{args.tag}.png"), "--metrics-json", str(art / f"eval_{args.tag}.json")]
    if args.randomize:
        cmd.append("--randomize")
    run(cmd, cwd=root)

    cmd = [sys.executable, "scripts/sweep_policies.py", "--policy", "torch", "--model", str(ppo_ckpt), "--episodes", "20", "--seed", str(args.seed), "--out", str(art / f"{args.tag}_sweep.csv")]
    if args.randomize:
        cmd.append("--randomize")
    run(cmd, cwd=root)

    cmd = [sys.executable, "scripts/render_animation.py", "--policy", "torch", "--model", str(ppo_ckpt), "--seed", str(args.seed), "--out", str(anim / f"{args.tag}.gif"), "--summary", str(anim / f"{args.tag}_summary.json")]
    if args.randomize:
        cmd.append("--randomize")
    run(cmd, cwd=root)

    cmd = [sys.executable, "scripts/build_policy_gallery.py", "--dir", str(anim), "--title", "SBW shared control v7 gallery", "--subtitle", f"History stack={args.history_stack}, randomized={args.randomize}"]
    run(cmd, cwd=root)


if __name__ == "__main__":
    main()

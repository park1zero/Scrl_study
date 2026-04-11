from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from bc_model import BCConfig, BCAuthorityActor, save_checkpoint


def train(dataset_path: Path, ckpt_path: Path, curve_path: Path, epochs: int = 20, batch_size: int = 256, lr: float = 1e-3, seed: int = 0) -> None:
    torch.set_num_threads(1)
    payload = np.load(dataset_path)
    obs = torch.tensor(payload["observations"], dtype=torch.float32)
    act = torch.tensor(payload["actions"], dtype=torch.float32)

    ds = TensorDataset(obs, act)
    n_val = max(1, int(0.15 * len(ds)))
    n_train = len(ds) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = BCAuthorityActor(BCConfig(input_dim=obs.shape[1]))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history = {"epoch": [], "train_loss": [], "val_loss": []}
    best_state = None
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for x, y in train_loader:
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                pred = model(x)
                loss = loss_fn(pred, y)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"epoch {epoch:02d} | train {train_loss:.6f} | val {val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    save_checkpoint(ckpt_path, model, extra={"input_dim": int(obs.shape[1]), "best_val_loss": best_val})

    curve_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(history["epoch"], history["train_loss"], label="train")
    plt.plot(history["epoch"], history["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.title("BC warm-start training curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="artifacts/teacher_dataset.npz")
    parser.add_argument("--out", type=str, default="artifacts/bc_actor.pt")
    parser.add_argument("--curve", type=str, default="artifacts/bc_training_curve.png")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    train(Path(args.dataset), Path(args.out), Path(args.curve), epochs=args.epochs)

"""Short, real comparative run: mixup on vs. off, same seed/architecture.

Deliberately NOT a full 30-epoch retrain -- this is a short run whose only
job is to show the mechanism actually trains and to report honest numbers
at that reduced epoch count. Mixup's benefit is a longer-training-run
phenomenon (it acts as a regularizer against overfitting, which mostly
shows up once a plain model starts memorizing); a short run may show a
small or inconclusive difference, and this script reports whatever it
actually measures rather than tuning epoch count to manufacture a bigger
gap. Writes to outputs/mixup_comparison.{json,png}, never touching the
existing outputs/history.json or outputs/training_curves.png (those are
the real, already-committed 30-epoch/90.55% run's evidence).
"""
import json
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import get_datasets
from model import CIFAR10CNN
from train import run_epoch

EPOCHS = 5
BATCH_SIZE = 128
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
VAL_SIZE = 5000
SEED = 42
NUM_WORKERS = 0
MIXUP_ALPHA = 0.4


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_short_training(mixup_alpha, device, train_set, val_set):
    torch.manual_seed(SEED)  # same init + same data order for both runs
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, drop_last=True,
                               generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=NUM_WORKERS)

    model = CIFAR10CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device,
                                           train=True, mixup_alpha=mixup_alpha)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"  [mixup_alpha={mixup_alpha}] epoch {epoch}/{EPOCHS}  "
              f"train_loss {train_loss:.4f}  train_acc {train_acc:.4f}  "
              f"val_loss {val_loss:.4f}  val_acc {val_acc:.4f}  ({time.time() - t0:.1f}s)")
    return history


def main():
    device = get_device()
    print(f"device: {device}")

    train_set, val_set, _ = get_datasets(val_size=VAL_SIZE, seed=SEED)
    print(f"train: {len(train_set)}  val: {len(val_set)}  (short comparison, {EPOCHS} epochs each)")

    print("\n--- baseline: no mixup ---")
    baseline = run_short_training(mixup_alpha=0.0, device=device, train_set=train_set, val_set=val_set)

    print(f"\n--- mixup: alpha={MIXUP_ALPHA} ---")
    mixup = run_short_training(mixup_alpha=MIXUP_ALPHA, device=device, train_set=train_set, val_set=val_set)

    result = {
        "epochs": EPOCHS,
        "mixup_alpha": MIXUP_ALPHA,
        "baseline": baseline,
        "mixup": mixup,
        "final_val_acc": {"baseline": baseline["val_acc"][-1], "mixup": mixup["val_acc"][-1]},
        "final_train_acc": {"baseline": baseline["train_acc"][-1], "mixup": mixup["train_acc"][-1]},
    }
    with open("outputs/mixup_comparison.json", "w") as f:
        json.dump(result, f, indent=2)

    epochs_range = range(1, EPOCHS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs_range, baseline["val_loss"], label="baseline val")
    axes[0].plot(epochs_range, mixup["val_loss"], label="mixup val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("val loss")
    axes[0].set_title(f"Val loss, {EPOCHS} epochs")
    axes[0].legend()

    axes[1].plot(epochs_range, baseline["val_acc"], label="baseline val")
    axes[1].plot(epochs_range, mixup["val_acc"], label="mixup val")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("val accuracy")
    axes[1].set_title(f"Val accuracy, {EPOCHS} epochs")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig("outputs/mixup_comparison.png", dpi=150)

    print(f"\nfinal val_acc -- baseline: {baseline['val_acc'][-1]:.4f}  mixup: {mixup['val_acc'][-1]:.4f}")
    print("saved outputs/mixup_comparison.json and outputs/mixup_comparison.png")


if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)
    main()

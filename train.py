import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import get_datasets
from model import CIFAR10CNN

EPOCHS = 30
BATCH_SIZE = 128
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
VAL_SIZE = 5000
SEED = 42
NUM_WORKERS = 4
CHECKPOINT_DIR = "outputs/checkpoints"


def parse_args():
    parser = argparse.ArgumentParser(description="Train the CIFAR-10 CNN.")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                         help=f"number of epochs to train (default: {EPOCHS})")
    parser.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_DIR,
                         help=f"directory to write resumable checkpoints (default: {CHECKPOINT_DIR})")
    return parser.parse_args()


def save_checkpoint(path, epoch, model, optimizer, scheduler, best_val_acc, history):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_acc": best_val_acc,
        "history": history,
    }, path)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total


def main():
    args = parse_args()
    num_epochs = args.epochs

    device = get_device()
    print(f"device: {device}")

    torch.manual_seed(SEED)

    train_set, val_set, test_set = get_datasets(val_size=VAL_SIZE, seed=SEED)
    print(f"train: {len(train_set)}  val: {len(val_set)}  test: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=NUM_WORKERS)

    model = CIFAR10CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    # SGD + momentum + weight decay is the classic recipe for small CIFAR-style
    # CNNs with BatchNorm; it generalizes at least as well as Adam here and is
    # what most reference CIFAR-10 training setups use, so it's the more
    # defensible/standard choice to explain in an interview.
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM,
                                 weight_decay=WEIGHT_DECAY)
    # Cosine annealing decays lr smoothly to ~0 over the full run with no
    # milestone epochs to hand-pick (unlike step decay). Since the epoch
    # budget is fixed and known upfront, cosine's smooth anneal-to-zero suits
    # a short, one-shot training run better than step decay's plateaus.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_acc = 0.0
    start_time = time.time()

    last_ckpt = os.path.join(args.checkpoint_dir, "last.pth")
    best_ckpt = os.path.join(args.checkpoint_dir, "best.pth")

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()
        epoch_time = time.time() - epoch_start

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        print(f"epoch {epoch:2d}/{num_epochs}  "
              f"train_loss {train_loss:.4f}  train_acc {train_acc:.4f}  "
              f"val_loss {val_loss:.4f}  val_acc {val_acc:.4f}  "
              f"lr {history['lr'][-1]:.5f}  ({epoch_time:.1f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "outputs/best_model.pth")
            save_checkpoint(best_ckpt, epoch, model, optimizer, scheduler, best_val_acc, history)

        save_checkpoint(last_ckpt, epoch, model, optimizer, scheduler, best_val_acc, history)

    total_time = time.time() - start_time
    print(f"training done in {total_time / 60:.1f} min. best val_acc: {best_val_acc:.4f}")

    history["total_time_sec"] = total_time
    history["best_val_acc"] = best_val_acc
    with open("outputs/history.json", "w") as f:
        json.dump(history, f, indent=2)

    epochs_range = range(1, num_epochs + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs_range, history["train_loss"], label="train")
    axes[0].plot(epochs_range, history["val_loss"], label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs_range, history["train_acc"], label="train")
    axes[1].plot(epochs_range, history["val_acc"], label="val")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig("outputs/training_curves.png", dpi=150)
    print("saved outputs/training_curves.png")


if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)
    main()

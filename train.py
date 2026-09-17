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
    parser.add_argument("--resume", type=str, default=None,
                         help="path to a checkpoint (e.g. outputs/checkpoints/last.pth) to resume training from")
    parser.add_argument("--patience", type=int, default=7,
                         help="stop training if val_acc doesn't improve for this many epochs (default: 7)")
    parser.add_argument("--mixup_alpha", type=float, default=0.0,
                         help="mixup interpolation strength (Beta(alpha, alpha)); 0 or unset disables mixup (default: 0.0)")
    return parser.parse_args()


def mixup_data(images, labels, alpha, device):
    """Mixes one batch with a random permutation of itself.

    Returns (mixed_images, labels_a, labels_b, lam) where labels_a are the
    original (unpermuted) labels and labels_b are the permuted-partner
    labels -- the loss is blended against BOTH via mixup_criterion below,
    rather than blending the labels themselves into soft one-hot targets.
    Blending labels and using plain cross-entropy on the blend is a common
    shortcut, but it's not what the mixup paper actually does and it isn't
    equivalent for cross-entropy (loss is not linear in a one-hot blend the
    way it is when you blend the two per-sample losses directly).
    """
    if alpha <= 0:
        return images, labels, labels, 1.0
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    index = torch.randperm(images.size(0), device=device)
    mixed_images = lam * images + (1 - lam) * images[index]
    return mixed_images, labels, labels[index], lam


def mixup_criterion(criterion, outputs, labels_a, labels_b, lam):
    return lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)


def save_checkpoint(path, epoch, num_epochs, model, optimizer, scheduler, best_val_acc, history):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "num_epochs": num_epochs,
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


def run_epoch(model, loader, criterion, optimizer, device, train, mixup_alpha=0.0):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            if train and mixup_alpha > 0:
                mixed_images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha, device)
                outputs = model(mixed_images)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            # Accuracy is measured against the original (unpermuted) label
            # even under mixup -- the input was blended, so "correct" here
            # is an approximation (agreement with the dominant/first label),
            # not a strict accuracy figure. See README.
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

    checkpoint = None
    if args.resume:
        # Load before building the scheduler: CosineAnnealingLR's T_max is
        # fixed at construction time, so resuming with a different --epochs
        # than the original run would silently desync the LR curve (the
        # restored scheduler state assumes the old T_max, which can make lr
        # climb back up instead of continuing to anneal). Keeping the
        # checkpoint's original epoch budget is what keeps the schedule
        # mathematically consistent across a resume.
        checkpoint = torch.load(args.resume, map_location=device)
        if checkpoint["num_epochs"] != num_epochs:
            print(f"note: checkpoint was trained toward --epochs {checkpoint['num_epochs']}; "
                  f"keeping that total for a consistent LR schedule (ignoring --epochs {num_epochs})")
            num_epochs = checkpoint["num_epochs"]
        model.load_state_dict(checkpoint["model_state_dict"])

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
    start_epoch = 1

    last_ckpt = os.path.join(args.checkpoint_dir, "last.pth")
    best_ckpt = os.path.join(args.checkpoint_dir, "best.pth")

    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_val_acc = checkpoint["best_val_acc"]
        history = checkpoint["history"]
        start_epoch = checkpoint["epoch"] + 1
        print(f"resumed from {args.resume}: completed epoch {checkpoint['epoch']}, "
              f"best_val_acc {best_val_acc:.4f}, continuing at epoch {start_epoch}")

    start_time = time.time()
    epochs_no_improve = 0

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device,
                                           train=True, mixup_alpha=args.mixup_alpha)
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
            epochs_no_improve = 0
            torch.save(model.state_dict(), "outputs/best_model.pth")
            save_checkpoint(best_ckpt, epoch, num_epochs, model, optimizer, scheduler, best_val_acc, history)
        else:
            epochs_no_improve += 1

        save_checkpoint(last_ckpt, epoch, num_epochs, model, optimizer, scheduler, best_val_acc, history)

        if epochs_no_improve >= args.patience:
            print(f"early stopping: val_acc hasn't improved for {epochs_no_improve} "
                  f"epochs (patience={args.patience}), stopping at epoch {epoch}")
            break

    total_time = time.time() - start_time
    print(f"training done in {total_time / 60:.1f} min. best val_acc: {best_val_acc:.4f}")

    history["total_time_sec"] = total_time
    history["best_val_acc"] = best_val_acc
    with open("outputs/history.json", "w") as f:
        json.dump(history, f, indent=2)

    epochs_range = range(1, len(history["train_loss"]) + 1)
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

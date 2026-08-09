import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from data import CIFAR10_MEAN, CIFAR10_STD, CLASSES, get_datasets
from model import CIFAR10CNN
from train import get_device


def unnormalize(img_tensor):
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    return (img_tensor * std + mean).clamp(0, 1)


def main():
    device = get_device()
    _, _, test_set = get_datasets()
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=4)

    model = CIFAR10CNN().to(device)
    model.load_state_dict(torch.load("outputs/best_model.pth", map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    test_acc = correct / total
    print(f"test accuracy: {test_acc:.4f}  ({correct}/{total})")

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    n_classes = len(CLASSES)
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for true_label, pred_label in zip(all_labels, all_preds):
        confusion[true_label, pred_label] += 1

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"CIFAR-10 confusion matrix (test acc {test_acc:.2%})")

    thresh = confusion.max() / 2
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, confusion[i, j], ha="center", va="center",
                     color="white" if confusion[i, j] > thresh else "black", fontsize=8)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig("outputs/confusion_matrix.png", dpi=150)
    print("saved outputs/confusion_matrix.png")

    # top confused pairs (excluding the diagonal) for the README write-up
    off_diag = confusion.copy()
    np.fill_diagonal(off_diag, 0)
    flat_idx = np.argsort(off_diag.ravel())[::-1][:10]
    print("top confused pairs (true -> predicted, count):")
    confusions = []
    for idx in flat_idx:
        i, j = np.unravel_index(idx, off_diag.shape)
        count = off_diag[i, j]
        if count == 0:
            continue
        print(f"  {CLASSES[i]:10s} -> {CLASSES[j]:10s}: {count}")
        confusions.append({"true": CLASSES[i], "pred": CLASSES[j], "count": int(count)})

    with open("outputs/test_results.json", "w") as f:
        json.dump({
            "test_accuracy": test_acc,
            "correct": correct,
            "total": total,
            "top_confusions": confusions,
        }, f, indent=2)

    # sample prediction grid: 8 correct + 8 incorrect, chosen deterministically
    rng = np.random.default_rng(42)
    correct_idx = np.where(all_preds == all_labels)[0]
    incorrect_idx = np.where(all_preds != all_labels)[0]
    sample_correct = rng.choice(correct_idx, size=min(8, len(correct_idx)), replace=False)
    sample_incorrect = rng.choice(incorrect_idx, size=min(8, len(incorrect_idx)), replace=False)
    sample_indices = list(sample_correct) + list(sample_incorrect)

    fig, axes = plt.subplots(4, 4, figsize=(11, 11))
    for ax, idx in zip(axes.flat, sample_indices):
        img, true_label = test_set[idx]
        pred_label = all_preds[idx]
        img = unnormalize(img).permute(1, 2, 0).numpy()
        ax.imshow(img)
        is_correct = true_label == pred_label
        color = "green" if is_correct else "red"
        ax.set_title(f"true: {CLASSES[true_label]}\npred: {CLASSES[pred_label]}",
                      color=color, fontsize=9)
        ax.axis("off")
    fig.suptitle("Sample predictions (top 2 rows correct, bottom 2 rows incorrect)")
    fig.tight_layout()
    fig.savefig("outputs/sample_predictions.png", dpi=150)
    print("saved outputs/sample_predictions.png")


if __name__ == "__main__":
    main()

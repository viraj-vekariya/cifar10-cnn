import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms

# Standard CIFAR-10 per-channel mean/std (computed over the training set),
# used everywhere rather than 0.5/0.5/0.5 because it centers/scales each
# channel to its actual statistics, which trains slightly faster and is the
# widely-used convention for this dataset.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def get_transforms():
    # Random crop with 4px zero-padding and random horizontal flip are the
    # standard CIFAR-10 augmentations: they cheaply simulate translation/
    # mirroring variance that the 50k training images don't otherwise cover.
    # CIFAR-10 is small (32x32, 50k images) so a plain CNN without this
    # augmentation memorizes the training set within a few epochs and its
    # val accuracy plateaus well below what augmented training reaches.
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return train_transform, eval_transform


def get_datasets(data_dir="./data", val_size=5000, seed=42):
    """Returns (train_subset, val_subset, test_dataset).

    CIFAR-10 ships with only train/test. We carve a validation split out of
    train for model selection (checkpointing on best val accuracy) so the
    test set stays untouched until final evaluation. The val split uses eval
    transforms (no augmentation) since it's meant to estimate generalization,
    not to be trained on -- so we instantiate the underlying dataset twice
    (once per transform) and share indices via Subset.
    """
    train_transform, eval_transform = get_transforms()

    full_train_aug = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    full_train_eval = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=eval_transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=eval_transform)

    n_total = len(full_train_aug)
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=generator).tolist()
    val_indices = perm[:val_size]
    train_indices = perm[val_size:]

    train_subset = Subset(full_train_aug, train_indices)
    val_subset = Subset(full_train_eval, val_indices)

    return train_subset, val_subset, test_dataset

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train import mixup_criterion, mixup_data


def test_disabled_mixup_returns_original_batch_unchanged():
    torch.manual_seed(0)
    images = torch.randn(6, 3, 32, 32)
    labels = torch.randint(0, 10, (6,))

    mixed, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.0, device="cpu")

    assert torch.equal(mixed, images)
    assert torch.equal(labels_a, labels)
    assert torch.equal(labels_b, labels)
    assert lam == 1.0


def test_mixup_lambda_is_a_valid_convex_combination_weight():
    torch.manual_seed(1)
    images = torch.randn(16, 3, 32, 32)
    labels = torch.randint(0, 10, (16,))

    for _ in range(20):
        _, _, _, lam = mixup_data(images, labels, alpha=0.4, device="cpu")
        assert 0.0 <= lam <= 1.0


def test_mixed_image_lies_between_the_two_originals_pointwise():
    torch.manual_seed(2)
    images = torch.rand(4, 3, 32, 32)  # rand (not randn): keeps values in [0, 1] so "between" is easy to check
    labels = torch.arange(4)

    mixed, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.4, device="cpu")

    # mixed[i] = lam * images[i] + (1 - lam) * images[permuted_i]; permuted_i
    # is recoverable from labels_b (labels_b[i] tells us which original
    # sample's label ended up paired with sample i).
    partner_index = labels_b.clone()  # labels are 0..3 == their own index here
    reconstructed = lam * images + (1 - lam) * images[partner_index]
    assert torch.allclose(mixed, reconstructed, atol=1e-6)

    lo = torch.minimum(images, images[partner_index])
    hi = torch.maximum(images, images[partner_index])
    assert torch.all(mixed >= lo - 1e-6)
    assert torch.all(mixed <= hi + 1e-6)
    assert torch.equal(labels_a, labels)


def test_mixup_criterion_reduces_to_plain_cross_entropy_when_lambda_is_one():
    torch.manual_seed(3)
    criterion = nn.CrossEntropyLoss()
    logits = torch.randn(10, 10)
    labels = torch.randint(0, 10, (10,))
    other_labels = torch.randint(0, 10, (10,))  # irrelevant when lam == 1.0

    plain_loss = criterion(logits, labels)
    blended_loss = mixup_criterion(criterion, logits, labels, other_labels, lam=1.0)

    assert torch.allclose(plain_loss, blended_loss)


def test_mixup_criterion_is_the_convex_combination_of_the_two_per_sample_losses():
    torch.manual_seed(4)
    criterion = nn.CrossEntropyLoss()
    logits = torch.randn(10, 10)
    labels_a = torch.randint(0, 10, (10,))
    labels_b = torch.randint(0, 10, (10,))
    lam = 0.3

    blended_loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
    expected = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)

    assert torch.allclose(blended_loss, expected)


def test_mixup_does_not_break_gradient_flow_through_the_model():
    torch.manual_seed(5)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
    criterion = nn.CrossEntropyLoss()
    images = torch.randn(4, 3, 32, 32)
    labels = torch.randint(0, 10, (4,))

    mixed, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.4, device="cpu")
    outputs = model(mixed)
    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} got a non-finite gradient"

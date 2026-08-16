import json

import pytest
import torch

from conftest import FakeRunEpoch, _tiny_dataset, run_main
import train as train_module
from model import CIFAR10CNN


def test_save_checkpoint_writes_expected_keys(tmp_path):
    model = CIFAR10CNN()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
    history = {"train_loss": [1.0], "train_acc": [0.1], "val_loss": [1.0], "val_acc": [0.2], "lr": [0.1]}

    path = tmp_path / "ckpt" / "last.pth"
    train_module.save_checkpoint(str(path), 1, 4, model, optimizer, scheduler, 0.2, history)

    assert path.exists()
    checkpoint = torch.load(path, map_location="cpu")
    assert checkpoint["epoch"] == 1
    assert checkpoint["num_epochs"] == 4
    assert checkpoint["best_val_acc"] == 0.2
    assert checkpoint["history"] == history
    assert set(checkpoint["model_state_dict"].keys()) == set(model.state_dict().keys())
    assert "param_groups" in checkpoint["optimizer_state_dict"]
    assert "T_max" in checkpoint["scheduler_state_dict"] or "base_lrs" in checkpoint["scheduler_state_dict"]


def test_full_run_creates_last_and_best_checkpoints(monkeypatch, isolated_train_env):
    isolated_train_env.script(
        train_results=[(1.0, 0.1), (0.9, 0.2), (0.8, 0.3)],
        val_results=[(1.0, 0.3), (0.9, 0.5), (1.0, 0.4)],
    )

    run_main(monkeypatch, ["--epochs", "3", "--patience", "10"])

    last = torch.load("outputs/checkpoints/last.pth", map_location="cpu")
    best = torch.load("outputs/checkpoints/best.pth", map_location="cpu")

    assert last["epoch"] == 3
    assert best["epoch"] == 2
    assert best["best_val_acc"] == pytest.approx(0.5)
    assert last["best_val_acc"] == pytest.approx(0.5)

    with open("outputs/history.json") as f:
        history = json.load(f)
    assert history["val_acc"] == [pytest.approx(v) for v in [0.3, 0.5, 0.4]]


def test_early_stopping_triggers_at_correct_epoch(monkeypatch, isolated_train_env):
    # val_acc improves at epoch 1 (0.5 > 0.0), then plateaus at 0.5 for
    # epochs 2 and 3 (not a strict improvement) — with patience=2 that's
    # two non-improving epochs in a row, so training must stop right after
    # epoch 3 and never call run_epoch for a 4th epoch.
    isolated_train_env.script(
        train_results=[(1.0, 0.1), (1.0, 0.1), (1.0, 0.1)],
        val_results=[(1.0, 0.5), (1.0, 0.5), (1.0, 0.5)],
    )

    run_main(monkeypatch, ["--epochs", "10", "--patience", "2"])

    with open("outputs/history.json") as f:
        history = json.load(f)
    assert len(history["val_acc"]) == 3

    last = torch.load("outputs/checkpoints/last.pth", map_location="cpu")
    assert last["epoch"] == 3


def test_resume_with_different_epochs_reproduces_reference_lr_schedule(monkeypatch, tmp_path_factory):
    def make_fake_get_datasets():
        return lambda val_size, seed: (
            _tiny_dataset(16, seed),
            _tiny_dataset(8, seed + 1),
            _tiny_dataset(8, seed + 2),
        )

    # Reference: a clean, uninterrupted 4-epoch run.
    ref_dir = tmp_path_factory.mktemp("reference")
    monkeypatch.chdir(ref_dir)
    fake = FakeRunEpoch()
    monkeypatch.setattr(train_module, "run_epoch", fake)
    monkeypatch.setattr(train_module, "get_datasets", make_fake_get_datasets())
    monkeypatch.setattr(train_module, "BATCH_SIZE", 8)
    monkeypatch.setattr(train_module, "NUM_WORKERS", 0)
    (ref_dir / "outputs").mkdir()
    fake.script(
        train_results=[(1.0, 0.1)] * 4,
        val_results=[(1.0, 0.3), (1.0, 0.4), (1.0, 0.5), (1.0, 0.6)],
    )
    run_main(monkeypatch, ["--epochs", "4", "--patience", "100"])
    with open("outputs/history.json") as f:
        reference_lr = json.load(f)["lr"]
    assert len(reference_lr) == 4

    # Phase 1: run toward --epochs 4 but let early stopping cut it short
    # after epoch 2 (val_acc improves at epoch 1, then doesn't at epoch 2,
    # and patience=1 stops right there) -- producing a real "interrupted"
    # checkpoint at epoch 2 with num_epochs=4 stored in it.
    phase1_dir = tmp_path_factory.mktemp("phase1")
    monkeypatch.chdir(phase1_dir)
    fake1 = FakeRunEpoch()
    monkeypatch.setattr(train_module, "run_epoch", fake1)
    monkeypatch.setattr(train_module, "get_datasets", make_fake_get_datasets())
    monkeypatch.setattr(train_module, "BATCH_SIZE", 8)
    monkeypatch.setattr(train_module, "NUM_WORKERS", 0)
    (phase1_dir / "outputs").mkdir()
    fake1.script(
        train_results=[(1.0, 0.1), (1.0, 0.1)],
        val_results=[(1.0, 0.5), (1.0, 0.4)],
    )
    run_main(monkeypatch, ["--epochs", "4", "--patience", "1"])
    with open("outputs/history.json") as f:
        assert len(json.load(f)["lr"]) == 2  # confirms it really did stop after epoch 2

    checkpoint_path = phase1_dir / "outputs" / "checkpoints" / "last.pth"
    assert checkpoint_path.exists()

    # Phase 2: resume that checkpoint but request a DIFFERENT --epochs (10
    # instead of the original 4) -- this is the exact scenario that used to
    # desync the cosine LR schedule. The fix keeps the checkpoint's original
    # num_epochs (4), so the resumed run should complete at epoch 4 and its
    # full lr trajectory should exactly match the uninterrupted reference.
    fake1.script(
        train_results=[(1.0, 0.1), (1.0, 0.1)],
        val_results=[(1.0, 0.55), (1.0, 0.6)],
    )
    run_main(monkeypatch, ["--epochs", "10", "--resume", str(checkpoint_path), "--patience", "100"])

    with open("outputs/history.json") as f:
        resumed_lr = json.load(f)["lr"]

    assert len(resumed_lr) == 4
    assert resumed_lr == [pytest.approx(v) for v in reference_lr]

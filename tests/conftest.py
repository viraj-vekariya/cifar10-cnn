import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import pytest
import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import train as train_module


class FakeRunEpoch:
    """Deterministic stand-in for train.run_epoch: pops pre-scripted
    (loss, acc) results instead of actually running the model, so tests
    control training/validation dynamics exactly and run in milliseconds
    with no real gradient descent.
    """

    def __init__(self):
        self.train_results = []
        self.val_results = []

    def script(self, train_results, val_results):
        self.train_results = list(train_results)
        self.val_results = list(val_results)

    def __call__(self, model, loader, criterion, optimizer, device, train):
        results = self.train_results if train else self.val_results
        if not results:
            raise AssertionError(
                "FakeRunEpoch ran out of scripted results — main() ran more "
                "epochs than the test expected (early stopping/resume logic "
                "likely didn't behave as intended)"
            )
        return results.pop(0)


def _tiny_dataset(n, seed):
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(n, 3, 32, 32, generator=generator)
    labels = torch.randint(0, 10, (n,), generator=generator)
    return TensorDataset(images, labels)


@pytest.fixture
def fake_run_epoch(monkeypatch):
    fake = FakeRunEpoch()
    monkeypatch.setattr(train_module, "run_epoch", fake)
    return fake


@pytest.fixture
def isolated_train_env(monkeypatch, tmp_path, fake_run_epoch):
    """Runs train.main() against a tiny synthetic dataset in an isolated
    tmp cwd, with run_epoch fully scripted: no real CIFAR-10 download, no
    real gradient descent, no writes outside tmp_path.
    """

    def fake_get_datasets(val_size, seed):
        return (
            _tiny_dataset(16, seed),
            _tiny_dataset(8, seed + 1),
            _tiny_dataset(8, seed + 2),
        )

    monkeypatch.setattr(train_module, "get_datasets", fake_get_datasets)
    monkeypatch.setattr(train_module, "BATCH_SIZE", 8)
    monkeypatch.setattr(train_module, "NUM_WORKERS", 0)
    monkeypatch.chdir(tmp_path)
    os.makedirs("outputs", exist_ok=True)
    return fake_run_epoch


def run_main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["train.py"] + argv)
    train_module.main()

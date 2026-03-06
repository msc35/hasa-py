"""End-to-end integration test: train a small MLP on synthetic data with label noise."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from hasa import HASA


class IndexedDataset(Dataset):
    """Wraps a TensorDataset to also return the integer index."""

    def __init__(self, dataset: TensorDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> tuple[int, torch.Tensor, torch.Tensor]:
        x, y = self.dataset[idx]
        return idx, x, y


def _make_noisy_dataset(
    n: int = 1000,
    noise_ratio: float = 0.3,
    seed: int = 42,
) -> tuple[IndexedDataset, torch.Tensor]:
    """Create a 2-class synthetic dataset with flipped labels.

    Returns the wrapped dataset and the *clean* labels for evaluation.
    """
    rng = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 20, generator=rng)
    w_true = torch.randn(20, generator=rng)
    clean_y = (x @ w_true > 0).long()

    noisy_y = clean_y.clone()
    n_flip = int(n * noise_ratio)
    flip_idx = torch.randperm(n, generator=rng)[:n_flip]
    noisy_y[flip_idx] = 1 - noisy_y[flip_idx]

    ds = TensorDataset(x, noisy_y)
    return IndexedDataset(ds), clean_y


class TestIntegration:
    """Integration test: HASA should outperform vanilla training under noise."""

    def test_hasa_improves_over_baseline(self) -> None:
        torch.manual_seed(0)
        n_samples = 800
        noise_ratio = 0.3
        epochs = 60
        window_size = 10
        lr = 1e-2

        dataset, clean_labels = _make_noisy_dataset(n=n_samples, noise_ratio=noise_ratio)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        # ---- Baseline (no selection) ----
        model_base = nn.Sequential(nn.Linear(20, 32), nn.ReLU(), nn.Linear(32, 2))
        opt_base = torch.optim.Adam(model_base.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(reduction="none")

        for _ in range(epochs):
            model_base.train()
            for _, x, y in loader:
                loss = criterion(model_base(x), y).mean()
                loss.backward()
                opt_base.step()
                opt_base.zero_grad()

        model_base.eval()
        with torch.no_grad():
            x_all = dataset.dataset.tensors[0]
            pred_base = model_base(x_all).argmax(dim=1)
            acc_base = (pred_base == clean_labels).float().mean().item()

        # ---- HASA ----
        torch.manual_seed(0)
        model_hasa = nn.Sequential(nn.Linear(20, 32), nn.ReLU(), nn.Linear(32, 2))
        opt_hasa = torch.optim.Adam(model_hasa.parameters(), lr=lr)
        selector = HASA(
            num_samples=n_samples,
            window_size=window_size,
            select_ratio=0.7,
        )

        for _ in range(epochs):
            model_hasa.train()
            for indices, x, y in loader:
                losses = criterion(model_hasa(x), y)
                mask = selector.step(indices, losses.detach())
                n_sel = mask.sum()
                loss = (losses * mask).sum() / n_sel if n_sel > 0 else losses.mean()
                loss.backward()
                opt_hasa.step()
                opt_hasa.zero_grad()
            selector.end_epoch()

        model_hasa.eval()
        with torch.no_grad():
            pred_hasa = model_hasa(x_all).argmax(dim=1)
            acc_hasa = (pred_hasa == clean_labels).float().mean().item()

        # HASA should achieve meaningfully better clean accuracy
        assert acc_hasa > acc_base - 0.05, (
            f"HASA acc ({acc_hasa:.3f}) should be close to or better "
            f"than baseline ({acc_base:.3f}) on clean labels"
        )

    def test_warm_up_trains_all_samples(self) -> None:
        """Verify that during warm-up every batch gets an all-True mask."""
        dataset, _ = _make_noisy_dataset(n=200)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        selector = HASA(num_samples=200, window_size=5, select_ratio=0.5)

        for epoch in range(5):
            for indices, x, _y in loader:
                losses = torch.rand(len(x))
                mask = selector.step(indices, losses)
                assert mask.all(), f"Warm-up epoch {epoch} should select all"
            selector.end_epoch()

    def test_hasa_trainer_wrapper(self) -> None:
        """Smoke test for the HASATrainer convenience class."""
        from hasa.callbacks import HASATrainer

        dataset, _ = _make_noisy_dataset(n=200)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        model = nn.Sequential(nn.Linear(20, 16), nn.ReLU(), nn.Linear(16, 2))
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss(reduction="none")
        selector = HASA(num_samples=200, window_size=3, select_ratio=0.8)

        trainer = HASATrainer(model, opt, criterion, selector)
        metrics = trainer.train_epoch(loader)

        assert "loss" in metrics
        assert "selected_frac" in metrics
        assert 0 < metrics["selected_frac"] <= 1.0

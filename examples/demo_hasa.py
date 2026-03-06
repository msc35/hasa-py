"""Quick demo: train a small MLP on synthetic data with 30% label noise using HASA.

Runs in ~10-30 seconds on CPU. Compares baseline (no selection) vs HASA.
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset

from hasa import HASA

N_TRAIN = 6000
N_TEST = 2000
N_CLASSES = 4
DIM = 15
NOISE_RATIO = 0.30
EPOCHS = 60
WINDOW = 10
SELECT_RATIO = 0.75
BATCH = 128
LR = 5e-3
WD = 1e-3
SEED = 7


class IndexedDataset(Dataset):
    def __init__(self, dataset: TensorDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[int, torch.Tensor, torch.Tensor]:
        x, y = self.dataset[idx]
        return idx, x, y


def make_data(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate moderately overlapping Gaussian clusters — hard enough that
    memorising noisy labels hurts, but clean labels are learnable."""
    rng = torch.Generator().manual_seed(seed)
    xs, ys = [], []
    n_per_class = n // N_CLASSES

    centers = torch.randn(N_CLASSES, DIM, generator=rng) * 2.0

    for c in range(N_CLASSES):
        blob = centers[c] + torch.randn(n_per_class, DIM, generator=rng) * 1.2
        xs.append(blob)
        ys.append(torch.full((n_per_class,), c, dtype=torch.long))

    return torch.cat(xs), torch.cat(ys)


def inject_noise(y: torch.Tensor, ratio: float, seed: int) -> torch.Tensor:
    """Flip labels uniformly to a *different* class (guaranteed wrong)."""
    rng = torch.Generator().manual_seed(seed)
    noisy = y.clone()
    n_flip = int(len(y) * ratio)
    flip_idx = torch.randperm(len(y), generator=rng)[:n_flip]
    for i in flip_idx:
        choices = [c for c in range(N_CLASSES) if c != y[i].item()]
        noisy[i] = choices[torch.randint(0, len(choices), (1,), generator=rng).item()]
    return noisy


def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(x).argmax(dim=1)
        return (pred == y).float().mean().item()


def make_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(DIM, 64), nn.ReLU(),
        nn.Linear(64, N_CLASSES),
    )


def train_run(
    use_hasa: bool,
    x_train: torch.Tensor, y_noisy: torch.Tensor,
    x_test: torch.Tensor, y_test: torch.Tensor,
) -> float:
    torch.manual_seed(SEED)

    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=WD, momentum=0.9)
    criterion = nn.CrossEntropyLoss(reduction="none")

    indexed_ds = IndexedDataset(TensorDataset(x_train, y_noisy))
    loader = DataLoader(indexed_ds, batch_size=BATCH, shuffle=True)

    selector = None
    if use_hasa:
        selector = HASA(
            num_samples=len(indexed_ds),
            window_size=WINDOW,
            select_ratio=SELECT_RATIO,
        )

    tag = "HASA" if use_hasa else "Baseline"

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for indices, x, y in loader:
            logits = model(x)
            losses = criterion(logits, y)

            if selector is not None:
                mask = selector.step(indices, losses.detach())
                n_sel = mask.sum()
                loss = (losses * mask).sum() / n_sel if n_sel > 0 else losses.mean()
            else:
                loss = losses.mean()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            n_batches += 1

        if selector is not None:
            selector.end_epoch()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            acc = evaluate(model, x_test, y_test)
            warmup_str = " (warmup)" if selector and selector.in_warmup else ""
            print(f"  [{tag:>8s}] Epoch {epoch+1:2d}/{EPOCHS}  "
                  f"loss={epoch_loss/n_batches:.4f}  clean_test_acc={acc:.4f}{warmup_str}")

    return evaluate(model, x_test, y_test)


def main() -> None:
    print("=" * 65)
    print("HASA Demo — Synthetic multi-class classification")
    print(f"  {N_TRAIN} train / {N_TEST} test, {N_CLASSES} classes, "
          f"{DIM}-dim, {NOISE_RATIO:.0%} label noise")
    print(f"  HASA: window={WINDOW}, select_ratio={SELECT_RATIO}, {EPOCHS} epochs")
    print("=" * 65)

    x_train, y_clean = make_data(N_TRAIN, seed=SEED)
    y_noisy = inject_noise(y_clean, NOISE_RATIO, seed=SEED + 1)
    x_test, y_test = make_data(N_TEST, seed=SEED + 100)

    n_flipped = (y_noisy != y_clean).sum().item()
    clean_baseline = evaluate(make_model(), x_test, y_test)
    print(f"\n  Labels flipped: {n_flipped}/{N_TRAIN} ({n_flipped/N_TRAIN:.1%})")

    t0 = time.time()

    print(f"\n--- Baseline (no selection) ---")
    acc_base = train_run(False, x_train, y_noisy, x_test, y_test)

    print(f"\n--- HASA (History-Aware Sampling) ---")
    acc_hasa = train_run(True, x_train, y_noisy, x_test, y_test)

    elapsed = time.time() - t0

    delta = acc_hasa - acc_base
    print(f"\n{'=' * 65}")
    print(f"  Baseline clean-test accuracy : {acc_base:.4f}")
    print(f"  HASA clean-test accuracy     : {acc_hasa:.4f}")
    print(f"  Improvement                  : {delta:+.4f}")
    print(f"  Total wall time              : {elapsed:.1f}s")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()

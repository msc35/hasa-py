# HASA — History-Aware Sampling Algorithm

[![Tests](https://github.com/selimb/hasa/actions/workflows/test.yml/badge.svg)](https://github.com/selimb/hasa/actions/workflows/test.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/hasa.svg)](https://pypi.org/project/hasa/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A lightweight, pip-installable PyTorch library for **robust deep learning training under label noise**.

HASA tracks each training sample's loss trajectory over a sliding window and uses **loss variance** as a noise indicator.
Clean samples stabilise quickly (low variance); noisy/mislabelled samples oscillate (high variance).
After a warm-up phase the algorithm masks out high-variance samples so that gradients are computed only from likely-clean data.

## Installation

```bash
pip install hasa
```

Or install directly from GitHub:

```bash
pip install git+https://github.com/selimb/hasa.git
```

For development:

```bash
git clone https://github.com/selimb/hasa.git
cd hasa
pip install -e ".[dev]"
```

## Quick Start

### 1. Wrap your dataset to return sample indices

HASA needs to map each loss value back to a specific dataset sample.
The simplest approach is an `IndexedDataset` wrapper:

```python
from torch.utils.data import Dataset

class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return idx, x, y
```

### 2. Train with HASA selection

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from hasa import HASA

dataset = IndexedDataset(my_dataset)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

model = MyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(reduction='none')  # MUST be unreduced

selector = HASA(num_samples=len(dataset), window_size=15, select_ratio=0.8)

for epoch in range(150):
    for indices, x, y in loader:
        logits = model(x)
        losses = criterion(logits, y)

        mask = selector.step(indices, losses.detach())

        # Divide by mask.sum() (not batch_size) to preserve gradient magnitude
        loss = (losses * mask).sum() / mask.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    selector.end_epoch()
```

### 3. (Optional) Langevin noise injection

Motivated by interpreting SGD as approximate Bayesian inference (Mandt, Hoffman & Blei, 2018):

```python
selector = HASA(
    num_samples=len(dataset),
    window_size=15,
    select_ratio=0.8,
    langevin_noise=1e-4,
)

# Inside the training loop, after optimizer.step():
selector.inject_langevin_noise(model)
```

## How It Works

1. **Warm-up phase** (first `window_size` epochs): all samples are used — the per-sample loss history buffer fills up.
2. **Selection phase** (after warm-up): for each batch, compute per-sample loss variance from the history buffer. Keep only the `select_ratio` fraction with the **lowest** variance. High-variance samples (likely mislabelled) are masked out.

```
              Warm-up (epochs 0..T-1)          Selection (epochs T+)
             ┌─────────────────────┐     ┌──────────────────────────────┐
  per-sample │  Record losses into │     │  Var(loss history) per sample│
  losses ───>│  ring buffer, train │────>│  Keep lowest-variance k%     │
             │  on ALL samples     │     │  Mask out the rest           │
             └─────────────────────┘     └──────────────────────────────┘
```

## Hyperparameters

| Parameter | Meaning | Tested Range | Default |
|-----------|---------|-------------|---------|
| `num_samples` | Total dataset size (for buffer allocation) | — | *required* |
| `window_size` | Loss values stored per sample (T) | 5, 10, 15 | 15 |
| `select_ratio` | Fraction of batch to keep (k) | 0.5 – 0.9 | 0.8 |
| `langevin_noise` | Scale of injected Gaussian noise | 0 or small | 0.0 |

## Checkpointing

HASA supports full state serialisation for resuming training:

```python
state = selector.state_dict()
torch.save(state, "hasa_checkpoint.pt")

# Restore later
selector.load_state_dict(torch.load("hasa_checkpoint.pt"))
```

## HASATrainer (convenience wrapper)

For a simpler API that handles the full training loop:

```python
from hasa.callbacks import HASATrainer

trainer = HASATrainer(model, optimizer, criterion, selector, device="cuda")
for epoch in range(num_epochs):
    metrics = trainer.train_epoch(dataloader)
    print(f"Epoch {epoch}: loss={metrics['loss']:.4f}, selected={metrics['selected_frac']:.2%}")
```

## Works with any PyTorch model

HASA is **model-agnostic** — it only looks at per-sample loss values. It works with any architecture (CNNs, Transformers, MLPs, etc.) and any loss function that supports `reduction='none'`:

- `nn.CrossEntropyLoss(reduction='none')` — classification
- `nn.MSELoss(reduction='none')` — regression
- `nn.BCEWithLogitsLoss(reduction='none')` — binary classification
- Any custom loss returning per-sample values

## API Reference

### `HASA(num_samples, window_size=15, select_ratio=0.8, langevin_noise=0.0, device="cpu")`

- **`step(sample_indices, losses) -> BoolTensor`** — update history, return selection mask.
- **`end_epoch()`** — advance the epoch counter. Must be called once per epoch.
- **`inject_langevin_noise(model)`** — add Gaussian noise to parameters.
- **`state_dict() / load_state_dict(d)`** — checkpoint support.
- **`epoch`** — current epoch (read-only property).
- **`in_warmup`** — True during the warm-up phase.

### `LossHistoryBuffer(num_samples, window_size, device)`

- **`update(indices, losses)`** — write losses into the ring buffer.
- **`variance(indices) -> Tensor`** — compute per-sample loss variance.
- **`is_ready(epoch) -> bool`** — True after warm-up.

### `hard_select(variances, select_ratio) -> BoolTensor`

Returns a mask keeping the lowest `select_ratio` fraction of samples by variance.

## Running Tests

```bash
pip install -e ".[dev]"
pytest --cov=hasa
```

## License

[MIT](LICENSE)

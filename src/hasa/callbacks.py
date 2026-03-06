"""Optional integrations for PyTorch Lightning and vanilla PyTorch loops.

These wrappers have **lazy imports** so that ``hasa`` never hard-depends on
PyTorch Lightning or any other framework beyond ``torch``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .selector import HASA

logger = logging.getLogger(__name__)


# ======================================================================
# PyTorch Lightning callback
# ======================================================================

def _get_lightning():  # noqa: ANN202
    """Lazy-import PyTorch Lightning."""
    try:
        import lightning.pytorch as pl  # lightning >= 2.0
    except ImportError:
        try:
            import pytorch_lightning as pl  # type: ignore[no-redef]
        except ImportError as exc:
            raise ImportError(
                "PyTorch Lightning is required for HASACallback. "
                "Install it with: pip install lightning"
            ) from exc
    return pl


class HASACallback:
    """PyTorch Lightning ``Callback`` that applies HASA sample selection.

    The callback intercepts ``on_train_batch_start`` to inject the HASA mask
    into the batch and ``on_train_epoch_end`` to advance the epoch counter.

    **Important:** your ``LightningDataModule`` must yield batches of the
    form ``(indices, x, y)`` where ``indices`` are dataset-level integer
    indices.  Your ``training_step`` must use ``reduction='none'`` for the
    loss and apply the mask that this callback stores on the trainer.

    Parameters
    ----------
    selector : HASA
        A pre-configured HASA selector instance.
    """

    def __init__(self, selector: HASA) -> None:
        pl = _get_lightning()
        # Dynamically subclass so isinstance checks pass
        self.__class__ = type(
            "HASACallback", (pl.Callback, self.__class__), {}
        )
        self.selector = selector

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        self.selector.end_epoch()

    def on_train_batch_start(
        self,
        trainer: Any,
        pl_module: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        # Store a reference so training_step can access the selector
        pl_module._hasa_selector = self.selector  # type: ignore[attr-defined]


# ======================================================================
# Vanilla PyTorch wrapper
# ======================================================================

class HASATrainer:
    """Thin wrapper that adds HASA selection to a standard PyTorch loop.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    optimizer : torch.optim.Optimizer
        Optimiser instance.
    criterion : nn.Module
        Loss function **with** ``reduction='none'``.
    selector : HASA
        Pre-configured HASA selector.
    device : str | torch.device
        Device to move data to.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        selector: HASA,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.selector = selector
        self.device = torch.device(device)

    def train_epoch(
        self,
        dataloader: DataLoader[Any],
        epoch_callback: Callable[[dict[str, float]], None] | None = None,
    ) -> dict[str, float]:
        """Run one training epoch with HASA selection.

        The dataloader must yield ``(indices, inputs, targets)``.

        Parameters
        ----------
        dataloader : DataLoader
            Must yield ``(indices, x, y)`` triples.
        epoch_callback : callable, optional
            Called at the end of the epoch with a metrics dict.

        Returns
        -------
        dict
            ``{"loss": <mean_selected_loss>, "selected_frac": <fraction>}``.
        """
        self.model.train()
        total_loss = 0.0
        total_selected = 0
        total_samples = 0

        for indices, x, y in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)
            indices = indices.to(self.device)

            logits = self.model(x)
            losses = self.criterion(logits, y)

            mask = self.selector.step(indices, losses.detach())
            n_selected = mask.sum().item()

            loss = (losses * mask).sum() / mask.sum() if n_selected > 0 else losses.mean()

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.selector.inject_langevin_noise(self.model)

            total_loss += loss.item() * int(n_selected if n_selected > 0 else losses.shape[0])
            total_selected += int(n_selected)
            total_samples += losses.shape[0]

        self.selector.end_epoch()

        metrics = {
            "loss": total_loss / max(total_selected, 1),
            "selected_frac": total_selected / max(total_samples, 1),
        }
        if epoch_callback is not None:
            epoch_callback(metrics)
        return metrics

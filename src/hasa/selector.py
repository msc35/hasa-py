"""HASA — History-Aware Sampling Algorithm selector."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from .buffer import LossHistoryBuffer
from .selection import hard_select

logger = logging.getLogger(__name__)


class HASA:
    """History-Aware Sampling Algorithm for robust training under label noise.

    HASA tracks each sample's loss trajectory over a sliding window and uses
    loss variance as a noise indicator: clean samples stabilise (low variance),
    while mislabelled samples oscillate (high variance).  After a warm-up
    phase the selector masks out high-variance (likely noisy) samples.

    Parameters
    ----------
    num_samples : int
        Total number of samples in the dataset.  Required for buffer
        allocation.
    window_size : int
        Number of past loss values stored per sample (T).
    select_ratio : float
        Fraction of each batch to keep during the selection phase, in
        ``(0, 1]``.  The lowest-variance ``select_ratio`` fraction is kept.
    langevin_noise : float
        Standard deviation of Gaussian noise injected into parameters after
        each optimiser step (Langevin dynamics).  Set to ``0.0`` to disable.
    device : str | torch.device
        Device for the history buffer.

    Examples
    --------
    >>> selector = HASA(num_samples=50000, window_size=15, select_ratio=0.8)
    >>> for epoch in range(num_epochs):
    ...     for indices, x, y in loader:
    ...         losses = criterion(model(x), y)          # (B,) unreduced
    ...         mask = selector.step(indices, losses.detach())
    ...         loss = (losses * mask).sum() / mask.sum()
    ...         loss.backward(); optimizer.step(); optimizer.zero_grad()
    ...     selector.end_epoch()
    """

    def __init__(
        self,
        num_samples: int,
        window_size: int = 15,
        select_ratio: float = 0.8,
        langevin_noise: float = 0.0,
        device: str | torch.device = "cpu",
    ) -> None:
        if not 0.0 < select_ratio <= 1.0:
            raise ValueError(f"select_ratio must be in (0, 1], got {select_ratio}")
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")

        self._num_samples = num_samples
        self._window_size = window_size
        self._select_ratio = select_ratio
        self._langevin_noise = langevin_noise
        self._device = torch.device(device)

        self._buffer = LossHistoryBuffer(num_samples, window_size, device=self._device)
        self._epoch: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def epoch(self) -> int:
        """Current epoch counter (0-based)."""
        return self._epoch

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def select_ratio(self) -> float:
        return self._select_ratio

    @property
    def num_samples(self) -> int:
        return self._num_samples

    @property
    def in_warmup(self) -> bool:
        """True while the buffer is still filling (first ``window_size`` epochs)."""
        return self._epoch < self._window_size

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def step(
        self,
        sample_indices: Tensor,
        losses: Tensor,
    ) -> Tensor:
        """Update history and return a sample-selection mask.

        Parameters
        ----------
        sample_indices : Tensor
            Shape ``(B,)`` int — dataset-level indices for the batch.
        losses : Tensor
            Shape ``(B,)`` float — unreduced per-sample losses.

        Returns
        -------
        Tensor
            Shape ``(B,)`` bool — ``True`` for samples that should
            contribute to the gradient.  During warm-up every sample is
            selected.
        """
        sample_indices = sample_indices.to(self._device)
        losses = losses.detach().to(self._device, dtype=torch.float32)

        self._buffer.update(sample_indices, losses)

        if self.in_warmup:
            return torch.ones(
                losses.shape[0], dtype=torch.bool, device=self._device
            )

        variances = self._buffer.variance(sample_indices)
        mask = hard_select(variances, self._select_ratio)
        return mask

    def end_epoch(self) -> None:
        """Advance the internal epoch counter.  Call once per epoch."""
        self._epoch += 1
        logger.debug("HASA epoch %d complete (warmup=%s)", self._epoch, self.in_warmup)

    # ------------------------------------------------------------------
    # Langevin noise injection
    # ------------------------------------------------------------------

    def inject_langevin_noise(self, model: nn.Module) -> None:
        """Add isotropic Gaussian noise to all model parameters.

        Motivated by interpreting SGD as approximate Bayesian inference
        (Mandt, Hoffman & Blei, 2018).  Only has an effect when
        ``langevin_noise > 0``.

        Parameters
        ----------
        model : nn.Module
            The model whose parameters will be perturbed.
        """
        if self._langevin_noise <= 0.0:
            return
        with torch.no_grad():
            for p in model.parameters():
                if p.requires_grad:
                    p.add_(torch.randn_like(p) * self._langevin_noise)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Return a serialisable snapshot of the full HASA state."""
        return {
            "num_samples": self._num_samples,
            "window_size": self._window_size,
            "select_ratio": self._select_ratio,
            "langevin_noise": self._langevin_noise,
            "epoch": self._epoch,
            "buffer": self._buffer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore HASA state from a previous ``state_dict``."""
        self._epoch = state["epoch"]
        self._select_ratio = state["select_ratio"]
        self._langevin_noise = state["langevin_noise"]
        self._buffer.load_state_dict(state["buffer"])

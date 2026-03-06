"""Per-sample loss history ring buffer for HASA."""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor

from .utils import ring_buffer_variance

logger = logging.getLogger(__name__)


class LossHistoryBuffer:
    """Fixed-size FIFO ring buffer that stores per-sample loss history.

    Parameters
    ----------
    num_samples : int
        Total number of samples in the dataset (N).
    window_size : int
        Number of loss values to keep per sample (T).
    device : str | torch.device
        Device on which to allocate the buffer.

    Attributes
    ----------
    buffer : Tensor
        Shape ``(num_samples, window_size)`` float32 storage.
    write_ptr : Tensor
        Shape ``(num_samples,)`` int64 — next write position per sample.
    fill_count : Tensor
        Shape ``(num_samples,)`` int64 — total writes per sample (saturates
        semantics handled by clamping during variance computation).
    """

    def __init__(
        self,
        num_samples: int,
        window_size: int,
        device: str | torch.device = "cpu",
    ) -> None:
        self.num_samples = num_samples
        self.window_size = window_size
        self.device = torch.device(device)

        self.buffer = torch.zeros(
            num_samples, window_size, dtype=torch.float32, device=self.device
        )
        self.write_ptr = torch.zeros(
            num_samples, dtype=torch.long, device=self.device
        )
        self.fill_count = torch.zeros(
            num_samples, dtype=torch.long, device=self.device
        )

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def update(self, indices: Tensor, losses: Tensor) -> None:
        """Write new loss values into the buffer for the given sample indices.

        Parameters
        ----------
        indices : Tensor
            Shape ``(B,)`` int64 — dataset indices.
        losses : Tensor
            Shape ``(B,)`` float32 — corresponding per-sample losses.
        """
        indices = indices.to(self.device)
        losses = losses.to(self.device, dtype=torch.float32)

        ptrs = self.write_ptr[indices]
        self.buffer[indices, ptrs] = losses

        self.write_ptr[indices] = (ptrs + 1) % self.window_size
        self.fill_count[indices] += 1

    def variance(self, indices: Tensor) -> Tensor:
        """Compute loss variance for the requested samples.

        Only uses filled entries; samples with ``fill_count < window_size``
        compute variance over the entries written so far.

        Parameters
        ----------
        indices : Tensor
            Shape ``(B,)`` int64 — dataset indices.

        Returns
        -------
        Tensor
            Shape ``(B,)`` — per-sample variance.
        """
        indices = indices.to(self.device)
        return ring_buffer_variance(
            self.buffer, self.fill_count, self.window_size, indices
        )

    def is_ready(self, epoch: int) -> bool:
        """Return True once the warm-up phase is complete.

        The buffer is considered ready after ``window_size`` epochs have
        elapsed (i.e. when ``epoch >= window_size``).

        Parameters
        ----------
        epoch : int
            Current (0-based) epoch index.
        """
        return epoch >= self.window_size

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Return a serialisable snapshot of the buffer state."""
        return {
            "num_samples": self.num_samples,
            "window_size": self.window_size,
            "buffer": self.buffer.cpu(),
            "write_ptr": self.write_ptr.cpu(),
            "fill_count": self.fill_count.cpu(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore buffer state from a previous ``state_dict``."""
        assert state["num_samples"] == self.num_samples, (
            f"num_samples mismatch: expected {self.num_samples}, "
            f"got {state['num_samples']}"
        )
        assert state["window_size"] == self.window_size, (
            f"window_size mismatch: expected {self.window_size}, "
            f"got {state['window_size']}"
        )
        self.buffer = state["buffer"].to(self.device)
        self.write_ptr = state["write_ptr"].to(self.device)
        self.fill_count = state["fill_count"].to(self.device)

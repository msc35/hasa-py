"""Utility helpers for HASA: variance computation, percentile, etc."""

from __future__ import annotations

import torch
from torch import Tensor


def ring_buffer_variance(
    buffer: Tensor,
    fill_counts: Tensor,
    window_size: int,
    indices: Tensor | None = None,
) -> Tensor:
    """Compute variance over filled entries of a ring buffer.

    Uses the identity Var(X) = E[X^2] - E[X]^2 for a single vectorised pass.

    Parameters
    ----------
    buffer : Tensor
        Shape ``(n_samples, window_size)``.
    fill_counts : Tensor
        Shape ``(n_samples,)`` — how many entries have been written per sample.
    window_size : int
        Maximum history length (T).
    indices : Tensor, optional
        If given, compute variance only for these rows.

    Returns
    -------
    Tensor
        Shape ``(len(indices),)`` or ``(n_samples,)`` — per-sample variance.
    """
    if indices is not None:
        buf = buffer[indices]
        counts = fill_counts[indices].clamp(max=window_size).float()
    else:
        buf = buffer
        counts = fill_counts.clamp(max=window_size).float()

    counts = counts.clamp(min=1)  # avoid division by zero
    mean = buf.sum(dim=1) / counts
    mean_sq = (buf ** 2).sum(dim=1) / counts
    var = (mean_sq - mean ** 2).clamp(min=0.0)
    return var


def masked_percentile(values: Tensor, ratio: float) -> Tensor:
    """Return the value at the ``ratio``-th quantile.

    Parameters
    ----------
    values : Tensor
        1-D tensor of values.
    ratio : float
        Quantile in (0, 1].  E.g. 0.8 → 80th percentile.

    Returns
    -------
    Tensor
        Scalar threshold.
    """
    return torch.quantile(values, ratio)

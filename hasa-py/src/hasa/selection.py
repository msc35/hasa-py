"""Selection logic for HASA — hard top-k threshold on variance."""

from __future__ import annotations

import torch
from torch import Tensor

from .utils import masked_percentile


def hard_select(variances: Tensor, select_ratio: float) -> Tensor:
    """Return a boolean mask keeping the lowest-variance samples.

    Parameters
    ----------
    variances : Tensor
        Shape ``(B,)`` — per-sample variance values for the current batch.
    select_ratio : float
        Fraction of samples to keep, in ``(0, 1]``.

    Returns
    -------
    Tensor
        Shape ``(B,)`` bool — True for kept (low-variance) samples.
    """
    if select_ratio >= 1.0:
        return torch.ones(variances.shape[0], dtype=torch.bool, device=variances.device)

    threshold = masked_percentile(variances, select_ratio)
    return variances <= threshold

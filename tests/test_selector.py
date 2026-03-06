"""Tests for hasa.selector.HASA and hasa.selection.hard_select."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from hasa import HASA, hard_select


class TestHardSelect:
    """Unit tests for the hard_select function."""

    def test_full_ratio_keeps_all(self) -> None:
        var = torch.tensor([1.0, 5.0, 3.0, 2.0])
        mask = hard_select(var, 1.0)
        assert mask.all()

    def test_half_ratio(self) -> None:
        var = torch.tensor([1.0, 5.0, 3.0, 2.0])
        mask = hard_select(var, 0.5)
        # 50th percentile threshold → keep samples with var ≤ median
        assert mask.sum().item() == 2
        assert mask[0].item()  # var=1.0, lowest
        assert mask[3].item()  # var=2.0, second lowest

    def test_output_is_bool(self) -> None:
        var = torch.rand(20)
        mask = hard_select(var, 0.7)
        assert mask.dtype == torch.bool


class TestHASA:
    """Tests for the main HASA selector."""

    def test_warmup_returns_all_true(self) -> None:
        """During the first window_size epochs, every sample is selected."""
        selector = HASA(num_samples=100, window_size=5, select_ratio=0.5)

        for epoch in range(5):
            indices = torch.arange(10)
            losses = torch.rand(10)
            mask = selector.step(indices, losses)
            assert mask.all(), f"Epoch {epoch}: expected all True during warm-up"
            selector.end_epoch()

    def test_selection_after_warmup(self) -> None:
        """After warm-up, high-variance samples get masked out."""
        T = 3
        selector = HASA(num_samples=10, window_size=T, select_ratio=0.5)
        indices = torch.arange(10)

        # Fill the buffer with T epochs of data
        for epoch in range(T):
            # Samples 0-4: stable losses (low variance)
            # Samples 5-9: oscillating losses (high variance)
            losses = torch.zeros(10)
            losses[:5] = 1.0 + 0.01 * epoch  # stable
            if epoch % 2 == 0:
                losses[5:] = 5.0
            else:
                losses[5:] = 0.1
            selector.step(indices, losses)
            selector.end_epoch()

        # Now in selection phase
        assert not selector.in_warmup

        losses_final = torch.zeros(10)
        losses_final[:5] = 1.0
        losses_final[5:] = 3.0
        mask = selector.step(indices, losses_final)

        # Low-variance samples (0-4) should mostly be selected
        assert mask[:5].sum().item() >= 3
        # High-variance samples (5-9) should mostly be masked
        assert mask[5:].sum().item() <= 3

    def test_epoch_counter(self) -> None:
        selector = HASA(num_samples=10, window_size=5)
        assert selector.epoch == 0
        selector.end_epoch()
        assert selector.epoch == 1
        for _ in range(9):
            selector.end_epoch()
        assert selector.epoch == 10

    def test_in_warmup_property(self) -> None:
        selector = HASA(num_samples=10, window_size=3)
        assert selector.in_warmup
        for _ in range(3):
            selector.end_epoch()
        assert not selector.in_warmup

    def test_invalid_select_ratio(self) -> None:
        with pytest.raises(ValueError, match="select_ratio"):
            HASA(num_samples=10, select_ratio=0.0)
        with pytest.raises(ValueError, match="select_ratio"):
            HASA(num_samples=10, select_ratio=1.5)

    def test_invalid_window_size(self) -> None:
        with pytest.raises(ValueError, match="window_size"):
            HASA(num_samples=10, window_size=0)

    def test_invalid_num_samples(self) -> None:
        with pytest.raises(ValueError, match="num_samples"):
            HASA(num_samples=0)

    def test_state_dict_roundtrip(self) -> None:
        selector = HASA(num_samples=20, window_size=5, select_ratio=0.7)
        indices = torch.arange(10)

        for _ in range(3):
            selector.step(indices, torch.rand(10))
            selector.end_epoch()

        state = selector.state_dict()

        selector2 = HASA(num_samples=20, window_size=5)
        selector2.load_state_dict(state)

        assert selector2.epoch == 3
        assert selector2.select_ratio == pytest.approx(0.7)
        assert torch.equal(
            selector._buffer.buffer, selector2._buffer.buffer
        )

    def test_langevin_noise_modifies_params(self) -> None:
        model = nn.Linear(10, 2)
        selector = HASA(num_samples=100, langevin_noise=0.1)

        params_before = [p.clone() for p in model.parameters()]
        selector.inject_langevin_noise(model)
        params_after = list(model.parameters())

        any_changed = any(
            not torch.equal(b, a) for b, a in zip(params_before, params_after)
        )
        assert any_changed, "Langevin noise should modify parameters"

    def test_no_langevin_when_zero(self) -> None:
        model = nn.Linear(10, 2)
        selector = HASA(num_samples=100, langevin_noise=0.0)

        params_before = [p.clone() for p in model.parameters()]
        selector.inject_langevin_noise(model)
        params_after = list(model.parameters())

        for b, a in zip(params_before, params_after):
            assert torch.equal(b, a), "No noise should be injected when ε=0"

    def test_step_detaches_losses(self) -> None:
        """Losses passed to step should not require grad tracking inside buffer."""
        selector = HASA(num_samples=10, window_size=3)
        indices = torch.arange(5)
        losses = torch.rand(5, requires_grad=True)
        selector.step(indices, losses)
        assert not selector._buffer.buffer.requires_grad

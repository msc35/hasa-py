"""Tests for hasa.buffer.LossHistoryBuffer."""

from __future__ import annotations

import torch
import pytest

from hasa.buffer import LossHistoryBuffer


class TestLossHistoryBuffer:
    """Core ring-buffer tests."""

    def test_basic_update_and_variance(self) -> None:
        buf = LossHistoryBuffer(num_samples=5, window_size=3)
        indices = torch.tensor([0, 1])
        losses = torch.tensor([1.0, 2.0])
        buf.update(indices, losses)

        assert buf.fill_count[0].item() == 1
        assert buf.fill_count[1].item() == 1
        assert buf.buffer[0, 0].item() == pytest.approx(1.0)
        assert buf.buffer[1, 0].item() == pytest.approx(2.0)

    def test_ring_buffer_overwrites_oldest(self) -> None:
        """Write T+2 values into a single slot; verify FIFO overwrite."""
        T = 3
        buf = LossHistoryBuffer(num_samples=1, window_size=T)

        values = [1.0, 2.0, 3.0, 4.0, 5.0]  # T + 2 values
        for v in values:
            buf.update(torch.tensor([0]), torch.tensor([v]))

        # After 5 writes with T=3 the ring should hold [4.0, 5.0, 3.0]
        # (ptr cycles: 0→1→2→0→1, so positions are [4,5,3])
        stored = buf.buffer[0].tolist()
        assert sorted(stored) == pytest.approx(sorted([4.0, 5.0, 3.0]))

    def test_variance_uses_only_recent_T_values(self) -> None:
        """Write T+5 values; variance must equal var of last T values."""
        T = 4
        buf = LossHistoryBuffer(num_samples=1, window_size=T)

        all_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        for v in all_values:
            buf.update(torch.tensor([0]), torch.tensor([v]))

        recent = torch.tensor(all_values[-T:], dtype=torch.float32)
        expected_var = recent.var(correction=0).item()

        computed_var = buf.variance(torch.tensor([0]))[0].item()
        assert computed_var == pytest.approx(expected_var, abs=1e-5)

    def test_partial_fill_variance(self) -> None:
        """Variance with fewer entries than window_size still works."""
        buf = LossHistoryBuffer(num_samples=2, window_size=10)
        buf.update(torch.tensor([0]), torch.tensor([3.0]))
        buf.update(torch.tensor([0]), torch.tensor([5.0]))

        var = buf.variance(torch.tensor([0]))[0].item()
        # var of [3, 5, 0, 0, ...] with fill_count=2 → E[X²]-E[X]²
        # but fill_count is clamped at 2 for the denominator,
        # so mean = (3+5)/2=4, mean_sq = (9+25)/2=17, var=17-16=1
        # However the buffer has zeros in the remaining slots which are
        # included in the sum. We sum all 10 slots but divide by
        # min(fill_count, window_size)=2.
        # sum = 3+5+0*8 = 8, sum_sq = 9+25+0 = 34
        # mean = 8/2 = 4, mean_sq = 34/2 = 17, var = 17-16 = 1
        assert var == pytest.approx(1.0, abs=1e-5)

    def test_is_ready(self) -> None:
        buf = LossHistoryBuffer(num_samples=1, window_size=5)
        for e in range(5):
            assert not buf.is_ready(e)
        assert buf.is_ready(5)
        assert buf.is_ready(100)

    def test_state_dict_roundtrip(self) -> None:
        buf = LossHistoryBuffer(num_samples=3, window_size=4)
        buf.update(torch.tensor([0, 2]), torch.tensor([1.5, 2.5]))
        buf.update(torch.tensor([0]), torch.tensor([3.0]))

        state = buf.state_dict()

        buf2 = LossHistoryBuffer(num_samples=3, window_size=4)
        buf2.load_state_dict(state)

        assert torch.equal(buf.buffer, buf2.buffer)
        assert torch.equal(buf.write_ptr, buf2.write_ptr)
        assert torch.equal(buf.fill_count, buf2.fill_count)

    def test_state_dict_mismatch_raises(self) -> None:
        buf = LossHistoryBuffer(num_samples=3, window_size=4)
        state = buf.state_dict()

        buf2 = LossHistoryBuffer(num_samples=5, window_size=4)
        with pytest.raises(AssertionError, match="num_samples mismatch"):
            buf2.load_state_dict(state)

    def test_multiple_samples_independent(self) -> None:
        """Updates to one sample don't affect another."""
        buf = LossHistoryBuffer(num_samples=3, window_size=5)
        buf.update(torch.tensor([0]), torch.tensor([10.0]))
        buf.update(torch.tensor([2]), torch.tensor([20.0]))

        assert buf.buffer[1].sum().item() == 0.0
        assert buf.fill_count[1].item() == 0

    def test_gpu_if_available(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        buf = LossHistoryBuffer(num_samples=10, window_size=5, device="cuda")
        indices = torch.tensor([0, 3, 7])
        losses = torch.tensor([1.0, 2.0, 3.0])
        buf.update(indices, losses)
        var = buf.variance(indices)
        assert var.device.type == "cuda"

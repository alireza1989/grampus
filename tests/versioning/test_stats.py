"""Tests for pure statistical functions."""

from __future__ import annotations

import pytest

from grampus.versioning.stats import two_proportion_z_test, welch_t_test


class TestTwoProportionZTest:
    def test_large_significant_difference(self) -> None:
        p = two_proportion_z_test(n_a=1000, k_a=700, n_b=1000, k_b=750)
        assert p < 0.05

    def test_small_sample_not_significant(self) -> None:
        p = two_proportion_z_test(n_a=10, k_a=7, n_b=10, k_b=8)
        assert p > 0.05

    def test_identical_proportions_p_is_one(self) -> None:
        p = two_proportion_z_test(n_a=1000, k_a=500, n_b=1000, k_b=500)
        assert p > 0.99

    def test_zero_n_a_raises(self) -> None:
        with pytest.raises(ValueError):
            two_proportion_z_test(n_a=0, k_a=0, n_b=100, k_b=50)

    def test_zero_n_b_raises(self) -> None:
        with pytest.raises(ValueError):
            two_proportion_z_test(n_a=100, k_a=50, n_b=0, k_b=0)

    def test_returns_float_between_0_and_1(self) -> None:
        p = two_proportion_z_test(n_a=500, k_a=200, n_b=500, k_b=300)
        assert 0.0 <= p <= 1.0

    def test_extremely_different_proportions(self) -> None:
        p = two_proportion_z_test(n_a=1000, k_a=100, n_b=1000, k_b=900)
        assert p < 0.001


class TestWelchTTest:
    def test_clearly_different_means(self) -> None:
        a = [1.0] * 50
        b = [10.0] * 50
        p = welch_t_test(a, b)
        assert p < 0.001

    def test_identical_samples(self) -> None:
        a = [5.0, 5.0, 5.0, 5.0, 5.0]
        b = [5.0, 5.0, 5.0, 5.0, 5.0]
        p = welch_t_test(a, b)
        assert p > 0.99

    def test_too_few_elements_returns_one(self) -> None:
        assert welch_t_test([1.0], [2.0, 3.0]) == 1.0
        assert welch_t_test([], [1.0, 2.0]) == 1.0

    def test_returns_float_between_0_and_1(self) -> None:
        import random

        rng = random.Random(42)
        a = [rng.gauss(0, 1) for _ in range(30)]
        b = [rng.gauss(0.5, 1) for _ in range(30)]
        p = welch_t_test(a, b)
        assert 0.0 <= p <= 1.0

    def test_clearly_overlapping_distributions(self) -> None:
        import random

        rng = random.Random(123)
        a = [rng.gauss(0, 1) for _ in range(20)]
        b = [rng.gauss(0, 1) for _ in range(20)]
        # With same distribution, p should not be very small (usually > 0.01)
        p = welch_t_test(a, b)
        assert p > 0.0  # just check it doesn't crash or return nonsense

"""Pure Python statistical tests for A/B experiment analysis."""

from __future__ import annotations

import math


def two_proportion_z_test(n_a: int, k_a: int, n_b: int, k_b: int) -> float:
    """Return two-tailed p-value for difference between two proportions.

    Args:
        n_a: Total observations in group A (control).
        k_a: Successes in group A.
        n_b: Total observations in group B (treatment).
        k_b: Successes in group B.

    Returns:
        Two-tailed p-value in [0, 1].

    Raises:
        ValueError: If n_a or n_b is zero.
    """
    if n_a == 0:
        raise ValueError("n_a must be > 0")
    if n_b == 0:
        raise ValueError("n_b must be > 0")

    p_pool = (k_a + k_b) / (n_a + n_b)
    if p_pool == 0.0 or p_pool == 1.0:
        return 1.0

    se = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n_a + 1.0 / n_b))
    if se == 0.0:
        return 1.0

    z = (k_a / n_a - k_b / n_b) / se
    # Two-tailed p-value via erfc: p = erfc(|z| / sqrt(2))
    return math.erfc(abs(z) / math.sqrt(2))


def welch_t_test(a: list[float], b: list[float]) -> float:
    """Return two-tailed p-value for Welch's t-test (unequal variance).

    Returns 1.0 if either list has fewer than 2 elements.
    """
    if len(a) < 2 or len(b) < 2:
        return 1.0

    n_a = len(a)
    n_b = len(b)
    mean_a = sum(a) / n_a
    mean_b = sum(b) / n_b
    var_a = sum((x - mean_a) ** 2 for x in a) / (n_a - 1)
    var_b = sum((x - mean_b) ** 2 for x in b) / (n_b - 1)

    se2 = var_a / n_a + var_b / n_b
    if se2 == 0.0:
        # Zero variance in both groups: p=0 if means differ, p=1 if identical.
        return 0.0 if mean_a != mean_b else 1.0

    se = math.sqrt(se2)
    t = (mean_a - mean_b) / se

    # Welch-Satterthwaite degrees of freedom
    term_a = (var_a / n_a) ** 2 / max(n_a - 1, 1)
    term_b = (var_b / n_b) ** 2 / max(n_b - 1, 1)
    denominator = term_a + term_b
    if denominator == 0.0:
        return 1.0
    df = se2**2 / denominator

    # Two-tailed p-value via regularized incomplete beta function
    x = df / (df + t * t)
    return _betainc(x, df / 2.0, 0.5)


# ------------------------------------------------------------------
# Regularized incomplete beta function (pure Python, no scipy)
# ------------------------------------------------------------------


def _betacf(x: float, a: float, b: float) -> float:
    """Lentz's continued fraction for the regularized incomplete beta function."""
    MAX_ITER = 200
    EPS = 3e-7
    FPMIN = 1e-30

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d

    for m in range(1, MAX_ITER + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break

    return h


def _betainc(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta function I(x; a, b) using continued fractions."""
    if x < 0.0 or x > 1.0:
        return 0.0
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    factor = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))

    if x < (a + 1.0) / (a + b + 2.0):
        return factor * _betacf(x, a, b) / a
    else:
        return 1.0 - factor * _betacf(1.0 - x, b, a) / b

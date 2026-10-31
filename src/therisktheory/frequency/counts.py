"""Claim count distributions: Poisson and negative binomial.

The number of claims ``N`` in a period takes the values ``0, 1, 2, ...``.
Kaas et al. (2008), Section 3.3, use the Poisson distribution for rare
events and the negative binomial when there is overdispersion,
``Var[N] > E[N]``. Parameterizations follow Table A (p. 368):

* ``Poisson(rate)``: ``Pr[N = n] = exp(-rate) rate**n / n!``.
* ``negative binomial(size, probability)`` with ``r = size`` and
  ``p = probability``: ``Pr[N = n] = C(r + n - 1, n) p**r (1 - p)**n``,
  eq. (3.74), p. 64; ``r`` need not be an integer.

The API mirrors the severity functions of ``severity_aggregate``:
``*_pmf``, ``*_cdf``, ``*_ppf`` and ``*_moments``, plus ``*_pgf`` and
``*_sample``. The probability generating function ``P_N(s) = E[s**N]``
accepts complex ``s``, so that the aggregate model can evaluate
``m_S(t) = P_N(m_X(t))`` and ``phi_S(t) = P_N(phi_X(t))`` (Kaas et al.,
2008, eq. (3.5), p. 43).

SciPy is imported inside the functions that need it, because importing
``scipy.stats`` is much slower than importing this module.

Reference:
    Kaas, R., Goovaerts, M., Dhaene, J. and Denuit, M. (2008). *Modern
    Actuarial Risk Theory: Using R*, 2nd ed. Springer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from therisktheory import _validation
from therisktheory.frequency import _common

if TYPE_CHECKING:
    import numpy.typing as npt

_COMPLEX_KINDS = frozenset("iufc")


# Poisson.


def poisson_count_pmf(
    n: npt.ArrayLike, *, rate: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Computes ``Pr[N = n]`` for ``N ~ Poisson(rate)``.

    Mathematical conditions:
        * ``rate >= 0`` and finite; ``rate = 0`` is the point mass at 0.
        * ``n`` is a whole number; negative values have probability 0.

    Args:
        n: Number of claims.
        rate: Poisson parameter ``lambda = E[N]``.

    Returns:
        The probabilities as a float64 array (0-d for scalar inputs).

    Raises:
        TypeError: If an input is not made of real numbers.
        ValueError: If ``n`` is not a whole number, ``rate`` is negative,
            NaN or infinite, or the shapes cannot be broadcast together.

    Examples:
        >>> poisson_count_pmf([0, 1, 2], rate=2.0).round(6)
        array([0.135335, 0.270671, 0.270671])
    """
    from scipy import stats

    k = _whole_numbers(n, "n")
    lam = _validation.as_nonnegative_finite_array(rate, "rate")
    _validation.check_broadcastable(n=k, rate=lam)
    return np.asarray(stats.poisson.pmf(k, lam), dtype=np.float64)


def poisson_count_cdf(
    x: npt.ArrayLike, *, rate: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Computes ``Pr[N <= x]`` for ``N ~ Poisson(rate)``.

    Args:
        x: Real argument; the cdf is a step function with jumps at
            ``0, 1, 2, ...``.
        rate: Poisson parameter ``lambda >= 0``.

    Returns:
        The cdf as a float64 array (0-d for scalar inputs).

    Raises:
        TypeError: If an input is not made of real numbers.
        ValueError: If ``x`` is NaN, ``rate`` is outside its domain, or the
            shapes cannot be broadcast together.

    Examples:
        >>> round(float(poisson_count_cdf(1.5, rate=2.0)), 6)
        0.406006
    """
    from scipy import stats

    t = _validation.as_real_array(x, "x", allow_inf=True)
    lam = _validation.as_nonnegative_finite_array(rate, "rate")
    _validation.check_broadcastable(x=t, rate=lam)
    return np.asarray(stats.poisson.cdf(np.floor(t), lam), dtype=np.float64)


def poisson_count_ppf(
    q: npt.ArrayLike, *, rate: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Computes the smallest ``n`` with ``Pr[N <= n] >= q``.

    Args:
        q: Probability level in ``[0, 1]``.
        rate: Poisson parameter ``lambda >= 0``.

    Returns:
        The quantiles as a float64 array; ``q = 1`` gives ``inf`` when
        ``rate > 0``.

    Raises:
        TypeError: If an input is not made of real numbers.
        ValueError: If ``q`` is outside ``[0, 1]``, ``rate`` is outside its
            domain, or the shapes cannot be broadcast together.

    Examples:
        >>> poisson_count_ppf([0.1, 0.5, 0.9], rate=2.0)
        array([0., 2., 4.])
    """
    from scipy import stats

    prob = _validation.as_probability_array(q, "q")
    lam = _validation.as_nonnegative_finite_array(rate, "rate")
    _validation.check_broadcastable(q=prob, rate=lam)
    result = np.asarray(stats.poisson.ppf(prob, lam), dtype=np.float64)
    # SciPy returns -1 at q = 0; the smallest value of the support is 0.
    return np.where(prob == 0, 0.0, result)


def poisson_count_moments(
    *, rate: npt.ArrayLike
) -> dict[str, npt.NDArray[np.float64]]:
    """Returns the first moments of ``N ~ Poisson(rate)``.

    All cumulants of the Poisson distribution equal ``lambda`` (Table A),
    so ``E[N] = Var[N] = E[(N - E[N])**3] = lambda``.

    Args:
        rate: Poisson parameter ``lambda >= 0``.

    Returns:
        A dict with float64 arrays ``mean``, ``variance``,
        ``second_raw_moment`` and ``third_central_moment``.

    Raises:
        TypeError: If ``rate`` is not made of real numbers.
        ValueError: If ``rate`` is negative, NaN or infinite.

    Examples:
        >>> poisson_count_moments(rate=3.0)["variance"]
        array(3.)
    """
    lam = _validation.as_nonnegative_finite_array(rate, "rate")
    return _common.pack_moments(mean=lam, variance=lam, third_central=lam)


def poisson_count_pgf(
    s: npt.ArrayLike, *, rate: npt.ArrayLike
) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128]:
    """Computes ``P_N(s) = E[s**N] = exp(lambda (s - 1))``.

    Args:
        s: Real or complex argument.
        rate: Poisson parameter ``lambda >= 0``.

    Returns:
        ``P_N(s)``, real for real ``s`` and complex for complex ``s``.

    Raises:
        TypeError: If ``s`` is not numeric or ``rate`` is not real.
        ValueError: If an input is NaN or infinite, or the shapes cannot be
            broadcast together.
        OverflowError: If the result exceeds the float64 range.

    Examples:
        >>> poisson_count_pgf(1.0, rate=2.0)
        array(1.)
    """
    z = _as_pgf_argument(s)
    lam = _validation.as_nonnegative_finite_array(rate, "rate")
    _validation.check_broadcastable(s=z, rate=lam)
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(np.exp(lam * (z - 1.0)))
    return _check_finite_pgf(result)


def poisson_count_sample(
    *,
    rate: float,
    sample_shape: int | tuple[int, ...],
    rng: np.random.Generator | int | None = None,
) -> npt.NDArray[np.int64]:
    """Draws claim counts from ``Poisson(rate)``.

    Args:
        rate: Poisson parameter ``lambda >= 0`` (scalar).
        sample_shape: Number or shape of the draws.
        rng: NumPy generator or seed, for reproducible results.

    Returns:
        An int64 array with the given shape.

    Raises:
        TypeError: If ``rate`` is not a real number.
        ValueError: If ``rate`` is not a finite non-negative scalar.
    """
    lam = _common.scalar(_validation.as_nonnegative_finite_array(rate, "rate"))
    generator = np.random.default_rng(rng)
    return generator.poisson(lam, size=sample_shape).astype(np.int64)


# Negative binomial.


def negative_binomial_count_pmf(
    n: npt.ArrayLike,
    *,
    size: npt.ArrayLike,
    probability: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Computes ``Pr[N = n] = C(r + n - 1, n) p**r (1 - p)**n``.

    Mathematical conditions:
        * ``r = size > 0`` and finite (not necessarily an integer).
        * ``0 < p = probability <= 1``; ``p = 1`` is the point mass at 0.
        * ``n`` is a whole number; negative values have probability 0.

    Args:
        n: Number of claims.
        size: Parameter ``r``.
        probability: Parameter ``p``.

    Returns:
        The probabilities as a float64 array (0-d for scalar inputs).

    Raises:
        TypeError: If an input is not made of real numbers.
        ValueError: If the conditions above do not hold or the shapes cannot
            be broadcast together.

    Examples:
        >>> negative_binomial_count_pmf([0, 1], size=2.0, probability=0.5)
        array([0.25, 0.25])
    """
    from scipy import stats

    k = _whole_numbers(n, "n")
    r, p = _negative_binomial_parameters(size, probability)
    _validation.check_broadcastable(n=k, size=r, probability=p)
    return np.asarray(stats.nbinom.pmf(k, r, p), dtype=np.float64)


def negative_binomial_count_cdf(
    x: npt.ArrayLike,
    *,
    size: npt.ArrayLike,
    probability: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Computes ``Pr[N <= x]`` for the negative binomial distribution.

    Args:
        x: Real argument.
        size: Parameter ``r > 0``.
        probability: Parameter ``0 < p <= 1``.

    Returns:
        The cdf as a float64 array (0-d for scalar inputs).

    Raises:
        TypeError: If an input is not made of real numbers.
        ValueError: If an input is outside its domain or the shapes cannot
            be broadcast together.

    Examples:
        >>> negative_binomial_count_cdf(1, size=2.0, probability=0.5)
        array(0.5)
    """
    from scipy import stats

    t = _validation.as_real_array(x, "x", allow_inf=True)
    r, p = _negative_binomial_parameters(size, probability)
    _validation.check_broadcastable(x=t, size=r, probability=p)
    return np.asarray(stats.nbinom.cdf(np.floor(t), r, p), dtype=np.float64)


def negative_binomial_count_ppf(
    q: npt.ArrayLike,
    *,
    size: npt.ArrayLike,
    probability: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Computes the smallest ``n`` with ``Pr[N <= n] >= q``.

    Args:
        q: Probability level in ``[0, 1]``.
        size: Parameter ``r > 0``.
        probability: Parameter ``0 < p <= 1``.

    Returns:
        The quantiles as a float64 array; ``q = 1`` gives ``inf`` when
        ``p < 1``.

    Raises:
        TypeError: If an input is not made of real numbers.
        ValueError: If an input is outside its domain or the shapes cannot
            be broadcast together.

    Examples:
        >>> negative_binomial_count_ppf(0.5, size=2.0, probability=0.5)
        array(1.)
    """
    from scipy import stats

    prob = _validation.as_probability_array(q, "q")
    r, p = _negative_binomial_parameters(size, probability)
    _validation.check_broadcastable(q=prob, size=r, probability=p)
    result = np.asarray(stats.nbinom.ppf(prob, r, p), dtype=np.float64)
    return np.where(prob == 0, 0.0, result)


def negative_binomial_count_moments(
    *, size: npt.ArrayLike, probability: npt.ArrayLike
) -> dict[str, npt.NDArray[np.float64]]:
    """Returns the first moments of the negative binomial distribution.

    With ``q = 1 - p`` (Table A, p. 368)::

        E[N] = r q / p,   Var[N] = r q / p**2 = E[N] / p,
        E[(N - E[N])**3] = r q (2 - p) / p**3.

    The overdispersion ``Var[N] / E[N] = 1 / p`` is at least 1 (Section
    3.9.4, p. 64).

    Args:
        size: Parameter ``r > 0``.
        probability: Parameter ``0 < p <= 1``.

    Returns:
        A dict with float64 arrays ``mean``, ``variance``,
        ``second_raw_moment`` and ``third_central_moment``.

    Raises:
        TypeError: If an input is not made of real numbers.
        ValueError: If an input is outside its domain or the shapes cannot
            be broadcast together.
        OverflowError: If a moment exceeds the float64 range.

    Examples:
        >>> m = negative_binomial_count_moments(size=2.0, probability=0.5)
        >>> m["mean"], m["variance"]
        (array(2.), array(4.))
    """
    r, p = _negative_binomial_parameters(size, probability)
    _validation.check_broadcastable(size=r, probability=p)
    q = 1.0 - p
    with np.errstate(over="ignore"):
        mean = np.asarray(r * q / p)
        variance = np.asarray(mean / p)
        third = np.asarray(variance * (2.0 - p) / p)
    for name, value in (
        ("mean", mean),
        ("variance", variance),
        ("third_central_moment", third),
    ):
        _validation.check_finite_result(value, name)
    return _common.pack_moments(
        mean=mean, variance=variance, third_central=third
    )


def negative_binomial_count_pgf(
    s: npt.ArrayLike,
    *,
    size: npt.ArrayLike,
    probability: npt.ArrayLike,
) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128]:
    """Computes ``P_N(s) = (p / (1 - (1 - p) s))**r``.

    The series ``E[s**N]`` converges for ``|(1 - p) s| < 1``; Kaas et al.
    (2008) give the mgf ``P_N(e**t)`` in Table A and eq. (3.15).

    Args:
        s: Real or complex argument with ``|(1 - p) s| < 1``.
        size: Parameter ``r > 0``.
        probability: Parameter ``0 < p <= 1``.

    Returns:
        ``P_N(s)``, real for real ``s`` and complex for complex ``s``.

    Raises:
        TypeError: If ``s`` is not numeric or a parameter is not real.
        ValueError: If the series diverges at ``s``, a parameter is outside
            its domain, or the shapes cannot be broadcast together.
        OverflowError: If the result exceeds the float64 range.

    Examples:
        >>> negative_binomial_count_pgf(1.0, size=2.0, probability=0.5)
        array(1.)
    """
    z = _as_pgf_argument(s)
    r, p = _negative_binomial_parameters(size, probability)
    _validation.check_broadcastable(s=z, size=r, probability=p)
    if np.any(np.abs((1.0 - p) * z) >= 1.0):
        raise ValueError("the pgf diverges: |(1 - probability) s| must be < 1")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray((p / (1.0 - (1.0 - p) * z)) ** r)
    return _check_finite_pgf(result)


def negative_binomial_count_sample(
    *,
    size: float,
    probability: float,
    sample_shape: int | tuple[int, ...],
    rng: np.random.Generator | int | None = None,
) -> npt.NDArray[np.int64]:
    """Draws claim counts from the negative binomial distribution.

    Args:
        size: Parameter ``r > 0`` (scalar).
        probability: Parameter ``0 < p <= 1`` (scalar).
        sample_shape: Number or shape of the draws.
        rng: NumPy generator or seed, for reproducible results.

    Returns:
        An int64 array with the given shape.

    Raises:
        TypeError: If a parameter is not a real number.
        ValueError: If a parameter is not a scalar inside its domain.
    """
    r, p = _negative_binomial_parameters(size, probability)
    generator = np.random.default_rng(rng)
    draws = generator.negative_binomial(
        _common.scalar(r), _common.scalar(p), size=sample_shape
    )
    return draws.astype(np.int64)


# Helpers.


def _negative_binomial_parameters(
    size: npt.ArrayLike, probability: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Validates ``r > 0`` and ``0 < p <= 1``."""
    r = _validation.as_nonnegative_finite_array(size, "size")
    if np.any(r == 0):
        raise ValueError("size must be positive")
    p = _validation.as_probability_array(probability, "probability")
    if np.any(p == 0):
        raise ValueError("probability must be positive")
    return r, p


def _whole_numbers(value: npt.ArrayLike, name: str) -> npt.NDArray[np.float64]:
    """Returns ``value`` as floats, checking that they are whole numbers."""
    array = _validation.as_real_array(value, name)
    if np.any(array != np.floor(array)):
        raise ValueError(f"{name} must contain whole numbers")
    return array


def _as_pgf_argument(
    value: npt.ArrayLike,
) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128]:
    """Accepts real or complex finite arguments for a pgf."""
    try:
        array = np.asarray(value)
    except ValueError as error:
        raise TypeError("s must be a number or array") from error
    if array.dtype.kind not in _COMPLEX_KINDS:
        raise TypeError(
            f"s must be a real or complex number, got {array.dtype}"
        )
    if array.dtype.kind == "c":
        array = array.astype(np.complex128, copy=False)
    else:
        array = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError("s must be finite")
    return array


def _check_finite_pgf(
    result: npt.NDArray[np.float64] | npt.NDArray[np.complex128],
) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128]:
    """Raises OverflowError if a pgf value is not finite."""
    if not np.all(np.isfinite(result)):
        raise OverflowError("pgf overflowed: the result is not finite")
    return result

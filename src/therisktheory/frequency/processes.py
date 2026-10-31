"""Claim number processes: constant risk, Pólya process and compound Poisson.

**Constant risk.** In a Poisson process with intensity ``lambda`` the
probability of one claim in ``(t, t + dt)`` is ``lambda dt`` whatever the
history (Kaas et al., 2008, Definition 4.2.1 and eq. (4.5), p. 89), and
``N(t) ~ Poisson(lambda t)``.

**Variable basic probabilities and the Pólya process.** If the intensity is
itself a random structure variable ``Lambda`` and, given ``Lambda``,
``N(t)`` is a Poisson process, the counts follow a mixed Poisson
distribution (Example 3.3.1, eqs. (3.13)-(3.15), pp. 45-46). With a gamma
structure distribution with mean ``lambda`` and shape ``h``, that is
``Lambda ~ gamma(h, h / lambda)``, this is the Pólya process:

* ``N(t)`` is negative binomial with ``r = h`` and
  ``p = h / (h + lambda t)``, because ``Lambda t ~ gamma(h, h / (lambda t))``
  and eq. (3.15) gives ``p = beta / (beta + 1)``;
* after ``k`` claims in ``(0, t)`` the posterior of ``Lambda`` is
  ``gamma(h + k, h / lambda + t)`` and the instantaneous claim rate is its
  mean, ``lambda (h + k) / (h + lambda t)`` (eq. (8.59), p. 225). Each claim
  raises the rate, which is the contagion effect of the Pólya process.

As ``h -> inf`` the gamma structure degenerates at ``lambda`` and the Pólya
process becomes the Poisson process; ``shape=numpy.inf`` is accepted for
that case.

**Compound Poisson process.** ``S(t) = X_1 + ... + X_{N(t)}`` with
``N(t) ~ Poisson(lambda t)``. Its cumulants are ``kappa_j = lambda t E[X**j]``
(Table A, p. 368), and a sum of independent compound Poisson variables is
again compound Poisson (Theorem 3.4.1, p. 47).

Reference:
    Kaas, R., Goovaerts, M., Dhaene, J. and Denuit, M. (2008). *Modern
    Actuarial Risk Theory: Using R*, 2nd ed. Springer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from therisktheory import _validation
from therisktheory.frequency import _common, counts

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt


def claim_intensity(
    elapsed_time: npt.ArrayLike,
    observed_claims: npt.ArrayLike,
    *,
    rate: npt.ArrayLike,
    shape: npt.ArrayLike = np.inf,
) -> npt.NDArray[np.float64]:
    """Computes the instantaneous claim rate given the history.

    With ``k`` claims observed in ``(0, t)``::

        lambda (h + k) / (h + lambda t)   (Pólya process, shape h)
        lambda                            (Poisson process, h = inf)

    Mathematical conditions:
        * ``t >= 0`` and ``k >= 0`` is a whole number.
        * ``lambda > 0`` is the prior mean intensity.
        * ``0 < h <= inf``; ``h = inf`` is constant risk.

    Args:
        elapsed_time: Observation time ``t``.
        observed_claims: Number of claims ``k`` observed in ``(0, t)``.
        rate: Prior mean intensity ``lambda = E[Lambda]``.
        shape: Shape ``h`` of the gamma structure distribution.

    Returns:
        The intensity as a float64 array.

    Raises:
        TypeError: If an input is not made of real numbers.
        ValueError: If the conditions above do not hold or the shapes cannot
            be broadcast together.

    Examples:
        >>> claim_intensity(2.0, 3, rate=0.5)
        array(0.5)
        >>> claim_intensity(2.0, 3, rate=0.5, shape=1.0)
        array(1.)
    """
    t = _validation.as_nonnegative_finite_array(elapsed_time, "elapsed_time")
    k = _validation.as_nonnegative_finite_array(
        observed_claims, "observed_claims"
    )
    if np.any(k != np.floor(k)):
        raise ValueError("observed_claims must contain whole numbers")
    lam = _positive(rate, "rate")
    h = _shape(shape)
    _validation.check_broadcastable(
        elapsed_time=t, observed_claims=k, rate=lam, shape=h
    )
    with np.errstate(invalid="ignore", over="ignore"):
        polya = lam * (h + k) / (h + lam * t)
        result = np.where(np.isinf(h), lam, polya)
    return _validation.check_finite_result(
        np.asarray(result, dtype=np.float64), "claim_intensity"
    )


def polya_process_parameters(
    *, time: npt.ArrayLike, rate: npt.ArrayLike, shape: npt.ArrayLike
) -> dict[str, npt.NDArray[np.float64]]:
    """Returns the negative binomial parameters of ``N(t)``.

    ``N(t)`` is negative binomial with ``size = h`` and
    ``probability = h / (h + lambda t)``.

    Args:
        time: Length ``t >= 0`` of the period.
        rate: Mean intensity ``lambda > 0``.
        shape: Finite shape ``h > 0`` of the gamma structure distribution.

    Returns:
        A dict with float64 arrays ``size`` and ``probability``, to be passed
        to the ``negative_binomial_count_*`` functions.

    Raises:
        TypeError: If an input is not made of real numbers.
        ValueError: If an input is outside its domain, ``shape`` is
            infinite, or the shapes cannot be broadcast together.

    Examples:
        >>> polya_process_parameters(time=2.0, rate=1.0, shape=2.0)
        {'size': array(2.), 'probability': array(0.5)}
    """
    t = _validation.as_nonnegative_finite_array(time, "time")
    lam = _positive(rate, "rate")
    h = _positive(shape, "shape")
    _validation.check_broadcastable(time=t, rate=lam, shape=h)
    h_b, t_b, lam_b = np.broadcast_arrays(h, t, lam)
    return {
        "size": np.asarray(h_b, dtype=np.float64),
        "probability": np.asarray(h_b / (h_b + lam_b * t_b), dtype=np.float64),
    }


def polya_process_count_pmf(
    n: npt.ArrayLike,
    *,
    time: npt.ArrayLike,
    rate: npt.ArrayLike,
    shape: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Computes ``Pr[N(t) = n]`` for the Pólya process.

    Args:
        n: Number of claims (whole number).
        time: Length ``t >= 0`` of the period.
        rate: Mean intensity ``lambda > 0``.
        shape: Shape ``h > 0`` of the gamma structure; ``inf`` gives the
            Poisson process.

    Returns:
        The probabilities as a float64 array.

    Raises:
        TypeError: If an input is not made of real numbers.
        ValueError: If an input is outside its domain or the shapes cannot
            be broadcast together.

    Examples:
        >>> polya_process_count_pmf(0, time=2.0, rate=1.0, shape=2.0)
        array(0.25)
    """
    h = _shape(shape)
    if np.all(np.isinf(h)):
        t = _validation.as_nonnegative_finite_array(time, "time")
        lam = _positive(rate, "rate")
        return counts.poisson_count_pmf(n, rate=lam * t)
    if np.any(np.isinf(h)):
        raise ValueError("shape must be all finite or all infinite")
    params = polya_process_parameters(time=time, rate=rate, shape=h)
    return counts.negative_binomial_count_pmf(n, **params)


def polya_process_moments(
    *, time: npt.ArrayLike, rate: npt.ArrayLike, shape: npt.ArrayLike
) -> dict[str, npt.NDArray[np.float64]]:
    """Returns the moments of ``N(t)`` for the Pólya process.

    By eq. (3.14), ``E[N(t)] = E[Lambda t] = lambda t`` and
    ``Var[N(t)] = E[Lambda t] + Var[Lambda t] = lambda t + (lambda t)**2 / h``.

    Args:
        time: Length ``t >= 0`` of the period.
        rate: Mean intensity ``lambda > 0``.
        shape: Shape ``h > 0``; ``inf`` gives the Poisson moments.

    Returns:
        A dict with float64 arrays ``mean``, ``variance``,
        ``second_raw_moment`` and ``third_central_moment``.

    Raises:
        TypeError: If an input is not made of real numbers.
        ValueError: If an input is outside its domain or the shapes cannot
            be broadcast together.

    Examples:
        >>> polya_process_moments(time=2.0, rate=1.0, shape=2.0)["variance"]
        array(4.)
    """
    t = _validation.as_nonnegative_finite_array(time, "time")
    lam = _positive(rate, "rate")
    h = _shape(shape)
    _validation.check_broadcastable(time=t, rate=lam, shape=h)
    mean = lam * t
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_h = np.where(np.isinf(h), 0.0, 1.0 / h)
    # Negative binomial cumulants with r = h and q / p = lambda t / h.
    variance = mean * (1.0 + mean * inv_h)
    third = mean * (1.0 + mean * inv_h) * (1.0 + 2.0 * mean * inv_h)
    return _common.pack_moments(
        mean=mean, variance=variance, third_central=third
    )


def polya_process_sample(
    *,
    time: float,
    rate: float,
    shape: float,
    sample_shape: int | tuple[int, ...],
    rng: np.random.Generator | int | None = None,
) -> npt.NDArray[np.int64]:
    """Simulates ``N(t)`` by drawing the intensity first.

    Each draw takes ``Lambda ~ gamma(h, h / lambda)`` and then
    ``N(t) ~ Poisson(Lambda t)``, which is the two-stage description of
    Example 3.3.1 and an independent check of the negative binomial form.

    Args:
        time: Length ``t >= 0`` of the period (scalar).
        rate: Mean intensity ``lambda > 0`` (scalar).
        shape: Finite shape ``h > 0`` (scalar).
        sample_shape: Number or shape of the draws.
        rng: NumPy generator or seed, for reproducible results.

    Returns:
        An int64 array of claim counts.

    Raises:
        TypeError: If an input is not a real number.
        ValueError: If an input is not a scalar inside its domain.
    """
    t = _common.scalar(_validation.as_nonnegative_finite_array(time, "time"))
    lam = _common.scalar(_positive(rate, "rate"))
    h = _common.scalar(_positive(shape, "shape"))
    generator = np.random.default_rng(rng)
    intensity = generator.gamma(h, lam / h, size=sample_shape)
    return generator.poisson(intensity * t).astype(np.int64)


def compound_poisson_cumulants(
    *,
    rate: npt.ArrayLike,
    severity_raw_moments: npt.ArrayLike,
    time: npt.ArrayLike = 1.0,
) -> npt.NDArray[np.float64]:
    """Computes the cumulants ``kappa_j = lambda t E[X**j]`` of ``S(t)``.

    The cumulant generating function of ``S(t)`` is
    ``lambda t (m_X(u) - 1)``, so its ``j``-th cumulant is ``lambda t`` times
    the ``j``-th raw moment of the claims. In particular ``E[S(t)]``,
    ``Var[S(t)]`` and ``E[(S(t) - E[S(t)])**3]`` are the first three.

    Args:
        rate: Intensity ``lambda >= 0``.
        severity_raw_moments: Raw moments ``E[X], E[X**2], ...`` along the
            last axis.
        time: Length ``t >= 0`` of the period.

    Returns:
        The cumulants with the same last axis as ``severity_raw_moments``.

    Raises:
        TypeError: If an input is not made of real numbers.
        ValueError: If an input is outside its domain or the shapes cannot
            be broadcast together.
        OverflowError: If a cumulant exceeds the float64 range.

    Examples:
        >>> compound_poisson_cumulants(rate=3.0, severity_raw_moments=[2, 8])
        array([ 6., 24.])
    """
    lam = _validation.as_nonnegative_finite_array(rate, "rate")
    t = _validation.as_nonnegative_finite_array(time, "time")
    mu = _validation.as_real_array(severity_raw_moments, "severity_raw_moments")
    if mu.ndim == 0 or mu.shape[-1] == 0:
        raise ValueError("severity_raw_moments must list at least one moment")
    if np.any(mu[..., 1::2] < 0):
        raise ValueError("even raw moments must be non-negative")
    scale = np.asarray(lam * t)[..., np.newaxis]
    _validation.check_broadcastable(scale=scale, severity_raw_moments=mu)
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(scale * mu, dtype=np.float64)
    return _validation.check_finite_result(result, "compound_poisson_cumulants")


def compound_poisson_merge(
    rates: npt.ArrayLike,
) -> dict[str, npt.NDArray[np.float64]]:
    """Combines independent compound Poisson portfolios (Theorem 3.4.1).

    If ``S_i`` is compound Poisson with intensity ``lambda_i`` and claims cdf
    ``P_i``, then ``S_1 + ... + S_m`` is compound Poisson with intensity
    ``lambda = sum_i lambda_i`` and claims cdf ``sum_i (lambda_i / lambda) P_i``
    (eq. (3.20), p. 47).

    Args:
        rates: Intensities ``lambda_i >= 0`` along the last axis, not all 0.

    Returns:
        A dict with ``rate`` (the total intensity) and ``weights`` (the
        mixing weights ``lambda_i / lambda`` of the claim distributions).

    Raises:
        TypeError: If ``rates`` is not made of real numbers.
        ValueError: If a rate is negative, NaN or infinite, or all are 0.

    Examples:
        >>> compound_poisson_merge([1.0, 3.0])
        {'rate': array(4.), 'weights': array([0.25, 0.75])}
    """
    lam = _validation.as_nonnegative_finite_array(rates, "rates")
    if lam.ndim == 0 or lam.shape[-1] == 0:
        raise ValueError("rates must list at least one portfolio")
    total = np.sum(lam, axis=-1)
    if np.any(total == 0):
        raise ValueError("at least one rate must be positive")
    total = _validation.check_finite_result(np.asarray(total), "rate")
    return {"rate": total, "weights": lam / total[..., np.newaxis]}


def compound_poisson_sample(
    *,
    rate: float,
    severity_sampler: Callable[
        [int, np.random.Generator], npt.NDArray[np.float64]
    ],
    sample_shape: int,
    time: float = 1.0,
    rng: np.random.Generator | int | None = None,
) -> npt.NDArray[np.float64]:
    """Simulates ``S(t)`` for a compound Poisson process.

    The claim counts are drawn first and all claim amounts are then drawn
    in a single call, so there is no Python loop over the simulations.

    Args:
        rate: Intensity ``lambda >= 0`` (scalar).
        severity_sampler: Function ``(n, generator) -> array`` returning
            ``n`` independent claim amounts.
        sample_shape: Number of simulated values of ``S(t)``.
        time: Length ``t >= 0`` of the period (scalar).
        rng: NumPy generator or seed, for reproducible results.

    Returns:
        A float64 array with ``sample_shape`` simulated totals.

    Raises:
        TypeError: If an input has the wrong type.
        ValueError: If an input is outside its domain or the sampler returns
            the wrong number of values.

    Examples:
        >>> def exp200(n, gen):
        ...     return gen.exponential(200.0, size=n)
        >>> s = compound_poisson_sample(
        ...     rate=3.0, severity_sampler=exp200, sample_shape=5, rng=1
        ... )
        >>> s.shape
        (5,)
    """
    lam = _common.scalar(_validation.as_nonnegative_finite_array(rate, "rate"))
    t = _common.scalar(_validation.as_nonnegative_finite_array(time, "time"))
    n_sims = int(sample_shape)
    if isinstance(sample_shape, bool) or n_sims != sample_shape or n_sims < 1:
        raise ValueError("sample_shape must be a positive integer")
    generator = np.random.default_rng(rng)
    claim_counts = generator.poisson(lam * t, size=n_sims)
    amounts = np.asarray(
        severity_sampler(int(claim_counts.sum()), generator), dtype=np.float64
    )
    if amounts.shape != (int(claim_counts.sum()),):
        raise ValueError("severity_sampler returned the wrong number of values")
    owners = np.repeat(np.arange(n_sims), claim_counts)
    return np.bincount(owners, weights=amounts, minlength=n_sims)


def _positive(value: npt.ArrayLike, name: str) -> npt.NDArray[np.float64]:
    """Validates a finite, strictly positive parameter."""
    array = _validation.as_nonnegative_finite_array(value, name)
    if np.any(array == 0):
        raise ValueError(f"{name} must be positive")
    return array


def _shape(value: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Validates ``0 < shape <= inf``."""
    array = _validation.as_real_array(value, "shape", allow_inf=True)
    if np.any(array <= 0):
        raise ValueError("shape must be positive")
    return array

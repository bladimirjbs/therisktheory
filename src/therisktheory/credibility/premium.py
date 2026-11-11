"""Credibility premium: construction, components and properties.

Credibility theory sets next period's premium for a risk (a contract or a
group of contracts) that has some claims experience of its own, inside a
portfolio with a lot more experience. Charging everyone the collective
mean ``m`` is right only if the portfolio is homogeneous; charging each risk
its own average ``X_j`` is right only if its experience is large enough. The
credibility premium is the compromise (Kaas et al., 2008, eq. (8.1),
p. 203)::

    P_j = z_j X_j + (1 - z_j) m,   0 <= z_j <= 1.

The credibility factor ``z_j`` measures how much the individual experience
is trusted. Each method of the course gives its own ``z_j``: Bühlmann
weights, limited fluctuation (partial and full credibility), Bayesian and
free-parameter credibility. This module combines any of them with the
individual and collective estimates, so the premium formula is written once.

In the balanced Bühlmann model the premium that minimizes the mean squared
error among linear predictors has this form with ``z = aT / (aT + s**2)``
(Theorems 8.2.2 and 8.2.4, pp. 207-209), and in the Bühlmann-Straub model
with ``z_j = a w_j / (s**2 + a w_j)`` (Theorem 8.4.1, p. 215). Computing
``z`` from the structural parameters is the job of the weight functions;
here ``z`` is an input.

Reference:
    Kaas, R., Goovaerts, M., Dhaene, J. and Denuit, M. (2008). *Modern
    Actuarial Risk Theory: Using R*, 2nd ed. Springer. Chapter 8.
    https://doi.org/10.1007/978-3-540-70998-5
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from therisktheory import _validation

if TYPE_CHECKING:
    import numpy.typing as npt

# Relative tolerance, in units of machine epsilon, for the bound check.
_BOUND_TOLERANCE_ULPS = 8


def credibility_premium(
    individual_mean: npt.ArrayLike,
    collective_mean: npt.ArrayLike,
    weight: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Computes the credibility premium ``z X_j + (1 - z) m``.

    Mathematical conditions:
        * ``0 <= z <= 1``; ``z = 0`` gives the collective premium ``m`` and
          ``z = 1`` the individual experience ``X_j``.
        * ``X_j`` and ``m`` are finite real numbers (claim amounts, loss
          ratios or claim frequencies, so they may be negative only if the
          quantity allows it).

    Args:
        individual_mean: Individual estimate ``X_j`` of the risk premium,
            usually the average claims of risk ``j``.
        collective_mean: Collective estimate ``m``: the overall mean, a
            known portfolio premium, or a weighted mean of the ``X_j``.
        weight: Credibility factor ``z``, for example from the Bühlmann
            weights.

    Returns:
        The premium as a float64 array; inputs are broadcast with the NumPy
        rules, and empty inputs give an empty array.

    Raises:
        TypeError: If an input is not made of real numbers.
        ValueError: If an input is NaN or infinite, ``z`` is outside
            ``[0, 1]``, or the shapes cannot be broadcast together.
        OverflowError: If the result exceeds the float64 range.

    Examples:
        >>> credibility_premium([100.0, 110.0, 120.0], 110.0, 0.782)
        array([102.18, 110.  , 117.82])
    """
    x, m, z = _inputs(individual_mean, collective_mean, weight)
    with np.errstate(over="ignore", invalid="ignore"):
        premium = np.asarray(z * x + (1.0 - z) * m, dtype=np.float64)
    return _validation.check_finite_result(premium, "credibility_premium")


def credibility_components(
    individual_mean: npt.ArrayLike,
    collective_mean: npt.ArrayLike,
    weight: npt.ArrayLike,
) -> dict[str, npt.NDArray[np.float64]]:
    """Splits the credibility premium into its components.

    The premium can be read as the collective premium plus a correction
    based on the individual experience::

        P_j = m + z (X_j - m).

    The correction is the part of the difference ``X_j - m`` that is
    believed; the rest, ``(1 - z)(X_j - m)``, is attributed to chance. The
    same premium also equals ``(k m + T X_j) / (k + T)`` with
    ``k = T (1 - z) / z``: the average claims after adding ``k`` periods of
    virtual experience at the level ``m`` (Remark 8.2.7, p. 211).

    Args:
        individual_mean: Individual estimate ``X_j``.
        collective_mean: Collective estimate ``m``.
        weight: Credibility factor ``z`` in ``[0, 1]``.

    Returns:
        A dict of float64 arrays with a common shape:

        * ``collective_premium``: ``m``;
        * ``individual_premium``: ``X_j``;
        * ``credibility_factor``: ``z``;
        * ``experience_correction``: ``z (X_j - m)``;
        * ``unexplained_deviation``: ``(1 - z)(X_j - m)``;
        * ``premium``: ``z X_j + (1 - z) m``.

    Raises:
        TypeError: If an input is not made of real numbers.
        ValueError: If an input is NaN or infinite, ``z`` is outside
            ``[0, 1]``, or the shapes cannot be broadcast together.
        OverflowError: If a component exceeds the float64 range.

    Examples:
        >>> c = credibility_components(120.0, 110.0, 0.782)
        >>> c["experience_correction"].round(2), c["premium"].round(2)
        (np.float64(7.82), np.float64(117.82))
    """
    x, m, z = np.broadcast_arrays(
        *_inputs(individual_mean, collective_mean, weight)
    )
    correction = _scaled_difference(z, x, m)
    unexplained = _scaled_difference(1.0 - z, x, m)
    for name, value in (
        ("experience_correction", correction),
        ("unexplained_deviation", unexplained),
    ):
        _validation.check_finite_result(value, name)
    return {
        "collective_premium": np.asarray(m, dtype=np.float64),
        "individual_premium": np.asarray(x, dtype=np.float64),
        "credibility_factor": np.asarray(z, dtype=np.float64),
        "experience_correction": correction,
        "unexplained_deviation": unexplained,
        "premium": credibility_premium(x, m, z),
    }


def credibility_premium_properties(
    individual_means: npt.ArrayLike,
    collective_mean: npt.ArrayLike,
    weights: npt.ArrayLike,
    *,
    exposures: npt.ArrayLike = 1.0,
    axis: int = -1,
) -> dict[str, npt.NDArray[np.float64] | npt.NDArray[np.bool_]]:
    """Checks the basic properties of a set of credibility premiums.

    **Between the two extremes.** Since ``0 <= z <= 1``, each premium lies
    between ``m`` and ``X_j``: credibility pulls every individual estimate
    towards the collective one (the shrinkage effect of Example 8.2.5,
    p. 210).

    **Portfolio balance.** With exposures ``c_j`` (for example the natural
    weights ``w_j`` of the Bühlmann-Straub model), the premiums reproduce the
    observed total, ``sum_j c_j P_j = sum_j c_j X_j``, exactly when::

        m = sum_j c_j (1 - z_j) X_j / sum_j c_j (1 - z_j),

    provided the denominator ``sum_j c_j (1 - z_j)`` is positive. If it is
    zero, every term ``c_j (1 - z_j)`` vanishes (for example, all risks with
    positive exposure have full credibility) and the portfolio is balanced
    for any ``m``. In the balanced model (equal ``z`` and ``c``) this ``m``
    is the overall mean (8.11), and in the Bühlmann-Straub model, where
    ``c_j (1 - z_j) = (s**2 / a) z_j``, it is the credibility weighted mean
    ``X_zw`` of (8.32) (Remark 8.4.4, p. 220). The derivation is in
    ``docs/topic_4/credibility_premium.md``.

    Args:
        individual_means: Individual estimates ``X_j`` along ``axis``.
        collective_mean: Collective estimate ``m`` (broadcast against the
            portfolio, so normally one value per portfolio).
        weights: Credibility factors ``z_j`` in ``[0, 1]``.
        exposures: Non-negative exposures ``c_j`` used for the balance.
        axis: Axis along which the risks of one portfolio are listed.

    Returns:
        A dict with:

        * ``premium``: the premiums ``P_j``;
        * ``lower_bound`` and ``upper_bound``: ``min`` and ``max`` of
          ``(X_j, m)``;
        * ``within_bounds``: booleans, true when the premium lies between
          them (allowing a few units of rounding);
        * ``portfolio_imbalance``: ``sum_j c_j (P_j - X_j)`` along
          ``axis``; zero means the premiums reproduce the observed total.

    Raises:
        TypeError: If an input is not made of real numbers.
        ValueError: If an input is outside its domain, ``individual_means``
            is a scalar, or the shapes cannot be broadcast together.
        OverflowError: If a result exceeds the float64 range.

    Examples:
        >>> means = [100.0, 110.0, 120.0]
        >>> props = credibility_premium_properties(means, 110.0, 0.782)
        >>> bool(props["within_bounds"].all())
        True
        >>> float(props["portfolio_imbalance"])
        0.0
    """
    x, m, z = _inputs(individual_means, collective_mean, weights)
    if x.ndim == 0:
        raise ValueError("individual_means must list the risks of a portfolio")
    c = _validation.as_nonnegative_finite_array(exposures, "exposures")
    _validation.check_broadcastable(
        individual_means=x, collective_mean=m, weights=z, exposures=c
    )
    x, m, z, c = np.broadcast_arrays(x, m, z, c)
    premium = credibility_premium(x, m, z)
    lower = np.minimum(x, m)
    upper = np.maximum(x, m)
    tol = _BOUND_TOLERANCE_ULPS * np.finfo(np.float64).eps
    with np.errstate(over="ignore"):  # comparing with +-inf is fine here
        slack = tol * np.maximum(np.abs(lower), np.abs(upper))
        within = (premium >= lower - slack) & (premium <= upper + slack)
    with np.errstate(over="ignore", invalid="ignore"):
        imbalance = np.asarray(np.sum(c * (premium - x), axis=axis))
    _validation.check_finite_result(imbalance, "portfolio_imbalance")
    return {
        "premium": premium,
        "lower_bound": np.asarray(lower, dtype=np.float64),
        "upper_bound": np.asarray(upper, dtype=np.float64),
        "within_bounds": np.asarray(within, dtype=np.bool_),
        "portfolio_imbalance": np.asarray(imbalance, dtype=np.float64),
    }


def _scaled_difference(
    factor: npt.NDArray[np.float64] | float,
    x: npt.NDArray[np.float64],
    m: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Computes ``factor * (x - m)`` without a spurious overflow.

    ``factor * (x - m)`` is the accurate form when ``x`` and ``m`` are
    close, because the subtraction is exact or nearly so. It overflows,
    however, when ``x - m`` exceeds the float64 range even though the
    scaled result does not, for example ``0.5 * (1e308 - (-1e308))``. Only
    there, where ``x`` and ``m`` have opposite signs and nothing cancels,
    the result is taken from ``factor * x - factor * m``, which overflows
    only if the true result does.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        deviation = x - m
        direct = factor * deviation
        expanded = factor * x - factor * m
    result = np.where(np.isfinite(deviation), direct, expanded)
    return np.asarray(result, dtype=np.float64)


def _inputs(
    individual_mean: npt.ArrayLike,
    collective_mean: npt.ArrayLike,
    weight: npt.ArrayLike,
) -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
]:
    """Validates the three inputs of the credibility premium."""
    x = _validation.as_real_array(individual_mean, "individual_mean")
    m = _validation.as_real_array(collective_mean, "collective_mean")
    z = _validation.as_probability_array(weight, "weight")
    _validation.check_broadcastable(
        individual_mean=x, collective_mean=m, weight=z
    )
    return x, m, z

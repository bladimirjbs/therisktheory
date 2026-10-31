"""Pure premium and the expected value premium principle.

Collective model (Kaas et al., 2008, Section 3.1, p. 41): ``N`` is the number
of claims in a period, ``X_1, X_2, ...`` are the claim amounts and
``S = X_1 + ... + X_N`` is the aggregate claim amount, with ``S = 0`` if
``N = 0``. The claim amounts are i.i.d. with the distribution of a random
variable ``X`` and are independent of ``N``.

Reference:
    Kaas, R., Goovaerts, M., Dhaene, J. and Denuit, M. (2008). *Modern
    Actuarial Risk Theory: Using R*, 2nd ed. Springer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from therisktheory import _validation

if TYPE_CHECKING:
    import numpy.typing as npt


def expected_aggregate_loss(
    frequency: npt.ArrayLike,
    mean_severity: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Computes the expected aggregate claims ``E[S] = E[N] * E[X]``.

    Mathematical conditions:
        * ``X_1, X_2, ...`` are i.i.d. and independent of ``N``.
        * ``0 <= E[N] < inf`` and ``0 <= E[X] < inf``.

    Under these conditions conditioning on ``N`` gives ``E[S | N] = N E[X]``
    and hence ``E[S] = E[N] E[X]`` (Kaas et al., 2008, eq. (3.3), p. 42).
    Only first moments are used, so the result holds for any claim count
    and claim size distributions with finite means.

    Args:
        frequency: Expected number of claims ``E[N]``.
        mean_severity: Expected claim amount ``E[X]``.

    Returns:
        ``E[S] >= 0`` as a float64 array (0-d for scalar inputs). The inputs
        are broadcast with the NumPy rules; empty inputs give an empty array.

    Raises:
        TypeError: If an input is not made of real numbers.
        ValueError: If an input is negative, NaN or infinite, or if the
            shapes cannot be broadcast together.
        OverflowError: If the product exceeds the float64 range.

    Examples:
        >>> expected_aggregate_loss(3.0, 200.0)
        array(600.)
        >>> expected_aggregate_loss([0.5, 2.0], 1000.0)
        array([ 500., 2000.])

    Note:
        Validation status: experimental.
    """
    freq = _validation.as_nonnegative_finite_array(frequency, "frequency")
    sev = _validation.as_nonnegative_finite_array(
        mean_severity, "mean_severity"
    )
    _validation.check_broadcastable(frequency=freq, mean_severity=sev)
    with np.errstate(over="ignore"):
        result = np.asarray(freq * sev)
    return _validation.check_finite_result(result, "expected_aggregate_loss")


def pure_premium(
    frequency: npt.ArrayLike,
    mean_severity: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Computes the pure (net) premium ``P = E[S] = E[N] * E[X]``.

    This is the net premium, or equivalence principle, of Kaas et al.
    (2008), Section 5.3, principle (a), p. 119. It has no safety loading;
    a premium without a positive loading leads to ruin with certainty
    (Section 5.3.1, p. 120).

    Mathematical conditions:
        The same as :func:`expected_aggregate_loss`.

    Args:
        frequency: Expected number of claims ``E[N]``.
        mean_severity: Expected claim amount ``E[X]``.

    Returns:
        The pure premium ``P >= 0`` as a float64 array (0-d for scalar
        inputs).

    Raises:
        TypeError: If an input is not made of real numbers.
        ValueError: If an input is negative, NaN or infinite, or if the
            shapes cannot be broadcast together.
        OverflowError: If the result exceeds the float64 range.

    Examples:
        >>> pure_premium(2.0, 500.0)
        array(1000.)

    Note:
        Validation status: experimental.
    """
    return expected_aggregate_loss(frequency, mean_severity)


def expected_value_premium(
    expected_loss: npt.ArrayLike,
    loading: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Computes the expected value premium ``(1 + theta) * E[S]``.

    Kaas et al. (2008), Section 5.3, principle (b), p. 119.

    Mathematical conditions:
        * ``0 <= E[S] < inf``.
        * ``0 <= theta < inf``. The book requires ``theta > 0``; ``theta = 0``
          is also accepted here and gives back the pure premium.

    Under these conditions the premium satisfies non-negative loading,
    ``(1 + theta) E[S] >= E[S]``, additivity and proportionality, but not
    no rip-off, consistency or iterativity when ``theta > 0`` (Section 5.3.1,
    pp. 120-121, and Table 5.1, p. 122). Iterativity asks that
    ``pi[pi[S | Y]] = pi[S]`` for every ``Y``; here the left side is
    ``(1 + theta)**2 E[S]`` by the tower property.

    Args:
        expected_loss: Expected aggregate claims ``E[S]``, for example the
            output of :func:`pure_premium`.
        loading: Relative safety loading ``theta``.

    Returns:
        The premium as a float64 array (0-d for scalar inputs).

    Raises:
        TypeError: If an input is not made of real numbers.
        ValueError: If an input is negative, NaN or infinite, or if the
            shapes cannot be broadcast together.
        OverflowError: If the result exceeds the float64 range.

    Examples:
        >>> expected_value_premium(1000.0, 0.2)
        array(1200.)

    Note:
        Validation status: experimental.
    """
    mean = _validation.as_nonnegative_finite_array(
        expected_loss, "expected_loss"
    )
    theta = _validation.as_nonnegative_finite_array(loading, "loading")
    _validation.check_broadcastable(expected_loss=mean, loading=theta)
    with np.errstate(over="ignore"):
        result = np.asarray((1.0 + theta) * mean)
    return _validation.check_finite_result(result, "expected_value_premium")

"""Helpers shared by the claim frequency modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from therisktheory import _validation

if TYPE_CHECKING:
    import numpy.typing as npt


def pack_moments(
    *,
    mean: npt.NDArray[np.float64],
    variance: npt.NDArray[np.float64],
    third_central: npt.NDArray[np.float64],
) -> dict[str, npt.NDArray[np.float64]]:
    """Packs the moments with the keys used by ``severity_aggregate``.

    Args:
        mean: ``E[N]``.
        variance: ``Var[N]``.
        third_central: ``E[(N - E[N])**3]``.

    Returns:
        A dict with ``mean``, ``variance``, ``second_raw_moment`` and
        ``third_central_moment`` as float64 arrays of a common shape.

    Raises:
        OverflowError: If ``E[N**2]`` exceeds the float64 range.
    """
    mean, variance, third_central = np.broadcast_arrays(
        mean, variance, third_central
    )
    with np.errstate(over="ignore"):
        second_raw = np.asarray(variance + mean**2, dtype=np.float64)
    _validation.check_finite_result(second_raw, "second_raw_moment")
    return {
        "mean": np.asarray(mean, dtype=np.float64),
        "variance": np.asarray(variance, dtype=np.float64),
        "second_raw_moment": second_raw,
        "third_central_moment": np.asarray(third_central, dtype=np.float64),
    }


def scalar(array: npt.NDArray[np.float64]) -> float:
    """Returns a 0-d array as a float.

    Args:
        array: Validated parameter.

    Returns:
        Its value as a Python float.

    Raises:
        ValueError: If ``array`` has more than one value.
    """
    if array.ndim != 0:
        raise ValueError("sampling parameters must be scalars")
    return float(array)

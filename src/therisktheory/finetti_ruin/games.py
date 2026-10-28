"""Fair and unfair games and the gambler's ruin problem.

A game pays the player a random amount ``Y`` in exchange for a stake ``c``.
The game is fair when the expected net gain ``E[Y] - c`` is zero, that is,
when the stake equals the mathematical expectation ``E[Y]``. Insurance at the
pure premium is a fair game in this sense; a loaded premium makes it unfair
for the insured.

In the gambler's ruin problem a player with ``z`` units bets one unit per
round, winning it with probability ``p`` and losing it with probability
``q = 1 - p``, until the fortune reaches ``0`` (ruin) or a target ``a``.
Letting ``a`` grow without bound gives a discrete-time risk process
``U(n) = u + G_1 + ... + G_n`` with ``G_i = +-1``, the model of Kaas et al.
(2008), Section 4.5, p. 98.

The ruin probability is derived step by step in
``docs/topic_2/gambler_ruin.md``.

Reference:
    Kaas, R., Goovaerts, M., Dhaene, J. and Denuit, M. (2008). *Modern
    Actuarial Risk Theory: Using R*, 2nd ed. Springer. Remark 4.4.3, p. 97,
    and Section 4.5, pp. 98-99.
"""

from __future__ import annotations

import dataclasses
import operator
from typing import TYPE_CHECKING

import numpy as np

from therisktheory import _validation

if TYPE_CHECKING:
    import numpy.typing as npt

_PROBABILITY_SUM_TOLERANCE = 1e-9
_SIMULATION_CHUNK = 256


def fair_game_expectation(
    probabilities: npt.ArrayLike,
    payoffs: npt.ArrayLike,
    *,
    axis: int = -1,
) -> npt.NDArray[np.float64]:
    """Computes the fair stake of a game, ``E[Y] = sum_i p_i y_i``.

    The game pays ``y_i`` with probability ``p_i``. Paying exactly ``E[Y]``
    makes the expected net gain zero, which is the definition of a fair game.

    Mathematical conditions:
        * ``0 <= p_i <= 1`` and ``sum_i p_i = 1`` along ``axis``.
        * The payoffs ``y_i`` are finite real numbers (they may be negative).

    Args:
        probabilities: Probabilities ``p_i`` of the outcomes.
        payoffs: Amounts ``y_i`` paid to the player in each outcome.
        axis: Axis along which the outcomes are listed. The other axes are
            treated as independent games and broadcast with NumPy rules.

    Returns:
        The fair stake ``E[Y]`` as a float64 array.

    Raises:
        TypeError: If an input is not made of real numbers.
        ValueError: If a probability is outside ``[0, 1]``, the
            probabilities do not add up to one, a payoff is NaN or infinite,
            or the shapes cannot be broadcast together.
        OverflowError: If the result exceeds the float64 range.

    Examples:
        >>> fair_game_expectation([0.5, 0.5], [0.0, 2.0])
        array(1.)
        >>> fair_game_expectation([0.9, 0.1], [0.0, 5000.0])
        array(500.)
    """
    probs, values = _outcome_arrays(probabilities, payoffs, axis)
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(np.sum(probs * values, axis=axis))
    return _validation.check_finite_result(result, "fair_game_expectation")


def unfair_game_expectation(
    probabilities: npt.ArrayLike,
    payoffs: npt.ArrayLike,
    stake: npt.ArrayLike,
    *,
    rounds: int = 1,
    axis: int = -1,
) -> npt.NDArray[np.float64]:
    """Computes the expected net gain ``n (E[Y] - c)`` of the player.

    Over ``n`` independent rounds with stake ``c`` the player's expected net
    gain is ``n (E[Y] - c)``. It is zero for a fair game, negative when the
    game favours the other party (for example, an insured paying a loaded
    premium) and positive when it favours the player.

    Mathematical conditions:
        * The same as :func:`fair_game_expectation`.
        * ``c`` is a finite real number and ``n >= 0`` is an integer.

    Args:
        probabilities: Probabilities ``p_i`` of the outcomes.
        payoffs: Amounts ``y_i`` paid to the player in each outcome.
        stake: Price ``c`` paid by the player each round.
        rounds: Number ``n`` of independent rounds.
        axis: Axis along which the outcomes are listed.

    Returns:
        The expected net gain as a float64 array.

    Raises:
        TypeError: If an input is not made of real numbers or ``rounds`` is
            not an integer.
        ValueError: If the conditions above do not hold or the shapes cannot
            be broadcast together.
        OverflowError: If the result exceeds the float64 range.

    Examples:
        >>> unfair_game_expectation([0.5, 0.5], [0.0, 2.0], stake=1.2)
        array(-0.2)
        >>> unfair_game_expectation([0.5, 0.5], [0.0, 2.0], 1.2, rounds=10)
        array(-2.)
    """
    n = _as_count(rounds, "rounds", minimum=0)
    price = _validation.as_real_array(stake, "stake")
    mean = fair_game_expectation(probabilities, payoffs, axis=axis)
    _validation.check_broadcastable(expectation=mean, stake=price)
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(n * (mean - price))
    return _validation.check_finite_result(result, "unfair_game_expectation")


def gambler_ruin_probability(
    initial_capital: npt.ArrayLike,
    target_capital: npt.ArrayLike,
    win_probability: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Computes the probability that the gambler is ruined.

    With ``r = q / p`` the probability of reaching ``0`` before ``a``,
    starting from ``z``, is::

        (r**z - r**a) / (1 - r**a)    if p != q,
        1 - z / a                     if p == q.

    It is the solution of ``P_z = p P_{z+1} + q P_{z-1}`` with ``P_0 = 1``
    and ``P_a = 0``, derived in ``docs/topic_2/gambler_ruin.md``. With an
    infinite target the probability of ever being ruined is ``r**z`` if
    ``p > q`` and ``1`` otherwise. In the notation of Kaas et al. (2008),
    Section 4.5, this is ``exp(-R z)`` with adjustment coefficient
    ``R = log(p / q)``.

    Numerically, each case is written so that no two nearly equal numbers
    are subtracted:

    * ``p > q``: ``r**z * (1 - r**(a - z)) / (1 - r**a)``, which keeps very
      small ruin probabilities (for example ``1e-29``) instead of rounding
      them to ``0``;
    * ``p < q``: numerator and denominator are divided by ``r**a``, which
      avoids overflow for large ``a``;
    * ``p = q``: ``(a - z) / a``, which is exact when ``z`` is close to ``a``.

    The differences ``1 - r**k`` are computed with ``expm1``, so the result
    stays accurate when ``p`` is close to ``1/2``.

    Mathematical conditions:
        * ``z`` and ``a`` are integers (measured in units of the stake) with
          ``0 <= z <= a``; ``a`` may be ``inf``.
        * ``0 <= p <= 1``; rounds are independent.

    Args:
        initial_capital: Initial fortune ``z``.
        target_capital: Fortune ``a`` at which the gambler stops, or
            ``numpy.inf`` to play until ruin.
        win_probability: Probability ``p`` of winning a round.

    Returns:
        The ruin probability, between 0 and 1, as a float64 array.

    Raises:
        TypeError: If an input is not made of real numbers.
        ValueError: If the conditions above do not hold or the shapes cannot
            be broadcast together.

    Examples:
        >>> gambler_ruin_probability(3, 10, 0.5)
        array(0.7)
        >>> round(float(gambler_ruin_probability(2, np.inf, 2 / 3)), 12)
        0.25
    """
    z = _validation.as_nonnegative_finite_array(
        initial_capital, "initial_capital"
    )
    a = _validation.as_real_array(
        target_capital, "target_capital", allow_inf=True
    )
    p = _validation.as_probability_array(win_probability, "win_probability")
    _validation.check_broadcastable(
        initial_capital=z, target_capital=a, win_probability=p
    )
    _check_integer(z, "initial_capital")
    _check_integer(a[np.isfinite(a)], "target_capital")
    if np.any(z > a):
        raise ValueError("initial_capital must not exceed target_capital")

    z, a, p = np.broadcast_arrays(z, a, p)
    finite = np.isfinite(a)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        # log1p(-p) is log(q) without rounding q = 1 - p first.
        log_r = np.log1p(-p) - np.log(p)  # +inf if p == 0, -inf if p == 1
        a_fin = np.where(finite, a, 0.0)
        fair = (a_fin - z) / np.where(a_fin > 0, a_fin, 1.0)
        # q > p (r > 1): divide numerator and denominator by r**a.
        q_wins = np.expm1(-(a_fin - z) * log_r) / np.expm1(-a_fin * log_r)
        # p > q (r < 1): factor out r**z; both expm1 terms are in (-1, 0).
        p_wins = (
            np.exp(z * log_r)
            * -np.expm1((a_fin - z) * log_r)
            / -np.expm1(a_fin * log_r)
        )
        no_target = np.where(log_r < 0, np.exp(z * log_r), 1.0)

    result = np.select(
        [
            z == 0,
            finite & (z == a),
            p == 0,
            p == 1,
            ~finite,
            log_r == 0,
            log_r > 0,
        ],
        [1.0, 0.0, 1.0, 0.0, no_target, fair, q_wins],
        default=p_wins,
    )
    return np.asarray(np.clip(result, 0.0, 1.0), dtype=np.float64)


@dataclasses.dataclass(frozen=True, slots=True)
class GamblerRuinSimulation:
    """Result of :func:`simulate_gambler_ruin`.

    Attributes:
        ruin_probability: Fraction of paths that reached ``0``.
        standard_error: Binomial standard error of ``ruin_probability``.
        mean_duration: Mean number of rounds of the paths that finished, or
            NaN if none finished.
        n_paths: Number of simulated paths.
        unfinished_paths: Paths still running after ``max_rounds``. They
            count as not ruined, so the estimate is then a lower bound.
    """

    ruin_probability: float
    standard_error: float
    mean_duration: float
    n_paths: int
    unfinished_paths: int


def simulate_gambler_ruin(
    initial_capital: int,
    target_capital: float,
    win_probability: float,
    *,
    n_paths: int = 10_000,
    max_rounds: int = 100_000,
    rng: np.random.Generator | int | None = None,
) -> GamblerRuinSimulation:
    """Estimates the gambler's ruin probability by Monte Carlo.

    All paths are simulated at once in blocks of rounds, so the only Python
    loop is over blocks while some path is still running. The result is an
    independent check of :func:`gambler_ruin_probability`.

    Args:
        initial_capital: Initial fortune ``z`` (integer, ``z >= 0``).
        target_capital: Target ``a`` (integer ``>= z``) or ``numpy.inf``.
        win_probability: Probability ``p`` of winning a round.
        n_paths: Number of simulated paths.
        max_rounds: Maximum number of rounds per path.
        rng: NumPy generator or seed, for reproducible results.

    Returns:
        A :class:`GamblerRuinSimulation` with the estimate.

    Raises:
        TypeError: If an input has the wrong type.
        ValueError: If the inputs do not satisfy the conditions of
            :func:`gambler_ruin_probability` or the counts are not positive.

    Examples:
        >>> sim = simulate_gambler_ruin(3, 10, 0.5, n_paths=20_000, rng=1)
        >>> abs(sim.ruin_probability - 0.7) < 4 * sim.standard_error
        True
    """
    n = _as_count(n_paths, "n_paths", minimum=1)
    max_n = _as_count(max_rounds, "max_rounds", minimum=1)
    inputs = (initial_capital, target_capital, win_probability)
    if any(np.ndim(x) != 0 for x in inputs):
        raise ValueError("simulate_gambler_ruin takes scalar inputs only")
    # Reuse the checks of the exact formula; scalars only.
    gambler_ruin_probability(initial_capital, target_capital, win_probability)
    z = float(initial_capital)
    a = float(target_capital)
    p = float(win_probability)
    generator = np.random.default_rng(rng)

    fortune = np.full(n, z)
    ruined = np.zeros(n, dtype=bool)
    duration = np.zeros(n, dtype=np.int64)
    if z <= 0:
        ruined[:] = True
        active = np.empty(0, dtype=np.int64)
    elif z >= a:
        active = np.empty(0, dtype=np.int64)
    else:
        active = np.arange(n)

    played = 0
    while active.size and played < max_n:
        block = min(_SIMULATION_CHUNK, max_n - played)
        steps = np.where(generator.random((active.size, block)) < p, 1, -1)
        paths = fortune[active, None] + np.cumsum(steps, axis=1)
        down = paths <= 0
        stop = down | (paths >= a)
        stopped = stop.any(axis=1)
        first = stop.argmax(axis=1)
        done = active[stopped]
        ruined[done] = down[stopped, first[stopped]]
        duration[done] = played + first[stopped] + 1
        fortune[active] = paths[:, -1]
        active = active[~stopped]
        played += block

    finished = np.ones(n, dtype=bool)
    finished[active] = False
    estimate = float(ruined.mean())
    return GamblerRuinSimulation(
        ruin_probability=estimate,
        standard_error=float(np.sqrt(estimate * (1.0 - estimate) / n)),
        mean_duration=(
            float(duration[finished].mean()) if finished.any() else np.nan
        ),
        n_paths=n,
        unfinished_paths=int(active.size),
    )


def _outcome_arrays(
    probabilities: npt.ArrayLike, payoffs: npt.ArrayLike, axis: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Validates the outcome probabilities and payoffs of a game."""
    probs = _validation.as_probability_array(probabilities, "probabilities")
    values = _validation.as_real_array(payoffs, "payoffs")
    _validation.check_broadcastable(probabilities=probs, payoffs=values)
    probs, values = np.broadcast_arrays(probs, values)
    if probs.ndim == 0:
        raise ValueError("probabilities must list at least one outcome")
    totals = np.sum(probs, axis=axis)
    if not np.allclose(totals, 1.0, rtol=0.0, atol=_PROBABILITY_SUM_TOLERANCE):
        raise ValueError("probabilities must add up to 1")
    return probs, values


def _as_count(value: int, name: str, *, minimum: int) -> int:
    """Returns ``value`` as an int, checking it is an integer >= minimum."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        count = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if count < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return count


def _check_integer(values: npt.NDArray[np.float64], name: str) -> None:
    """Raises ValueError if some value is not a whole number."""
    if np.any(values != np.floor(values)):
        raise ValueError(f"{name} must be a whole number of stakes")

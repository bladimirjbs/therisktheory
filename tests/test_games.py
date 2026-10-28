import doctest
import fractions
import warnings

import numpy as np
import pytest

from therisktheory.finetti_ruin import games
from therisktheory.solvency_pricing import premiums

fair_game_expectation = games.fair_game_expectation
unfair_game_expectation = games.unfair_game_expectation
gambler_ruin_probability = games.gambler_ruin_probability
simulate_gambler_ruin = games.simulate_gambler_ruin


def solve_first_step(a: int, p: float, constant: float, at_zero: float):
    """Solves x_k = c + p x_{k+1} + q x_{k-1}, x_0 = at_zero, x_a = 0.

    With c = 0 and x_0 = 1 this is the ruin probability; with c = 1 and
    x_0 = 0 it is the expected number of rounds until the game ends. Both
    come from conditioning on the first round.
    """
    q = 1.0 - p
    matrix = np.eye(a + 1)
    rhs = np.zeros(a + 1)
    rhs[0] = at_zero
    for k in range(1, a):
        matrix[k, k + 1] = -p
        matrix[k, k - 1] = -q
        rhs[k] = constant
    return np.linalg.solve(matrix, rhs)


def ruin_by_linear_system(z: int, a: int, p: float) -> float:
    return float(solve_first_step(a, p, constant=0.0, at_zero=1.0)[z])


def expected_duration(z: int, a: int, p: float) -> float:
    return float(solve_first_step(a, p, constant=1.0, at_zero=0.0)[z])


def test_docstring_examples() -> None:
    result = doctest.testmod(games)
    assert result.attempted > 0
    assert result.failed == 0


# Fair and unfair games.


def test_fair_game_expectation_is_the_mean_payoff() -> None:
    assert fair_game_expectation([0.2, 0.5, 0.3], [0, 10, 20]).item() == 11.0


def test_fair_game_expectation_along_axis() -> None:
    probs = [[0.5, 0.5], [0.9, 0.1]]
    payoffs = [[0.0, 2.0], [0.0, 5000.0]]
    np.testing.assert_allclose(
        fair_game_expectation(probs, payoffs), [1.0, 500.0]
    )
    np.testing.assert_allclose(
        fair_game_expectation(
            np.transpose(probs), np.transpose(payoffs), axis=0
        ),
        [1.0, 500.0],
    )


def test_negative_payoffs_are_allowed() -> None:
    assert fair_game_expectation([0.5, 0.5], [-1.0, 1.0]).item() == 0.0


def test_fair_stake_gives_zero_expected_gain() -> None:
    probs, payoffs = [0.3, 0.7], [100.0, 40.0]
    stake = fair_game_expectation(probs, payoffs)
    assert unfair_game_expectation(probs, payoffs, stake).item() == 0.0


def test_unfair_game_expectation_signs() -> None:
    probs, payoffs = [0.5, 0.5], [0.0, 2.0]
    assert unfair_game_expectation(probs, payoffs, 1.5).item() < 0
    assert unfair_game_expectation(probs, payoffs, 0.5).item() > 0


def test_unfair_game_scales_with_rounds() -> None:
    one = unfair_game_expectation([0.5, 0.5], [0.0, 2.0], 1.1)
    many = unfair_game_expectation([0.5, 0.5], [0.0, 2.0], 1.1, rounds=25)
    np.testing.assert_allclose(many, 25 * one)
    assert unfair_game_expectation([1.0], [3.0], 1.0, rounds=0).item() == 0.0


def test_unfair_game_broadcasts_over_stakes() -> None:
    result = unfair_game_expectation([0.5, 0.5], [0.0, 2.0], [0.5, 1.0, 1.5])
    np.testing.assert_allclose(result, [0.5, 0.0, -0.5])


def test_insurance_with_loaded_premium_is_unfair_for_the_insured() -> None:
    # Integration with Topic 1: the insured pays the expected value premium
    # and receives the claim. Expected net gain is -theta * E[S].
    probs = np.array([0.9, 0.08, 0.02])
    claims = np.array([0.0, 1000.0, 20000.0])
    theta = 0.15
    mean_claim = fair_game_expectation(probs, claims)
    premium = premiums.expected_value_premium(mean_claim, theta)
    gain = unfair_game_expectation(probs, claims, premium)
    np.testing.assert_allclose(gain, -theta * mean_claim)
    np.testing.assert_allclose(
        unfair_game_expectation(probs, claims, mean_claim), 0.0
    )


@pytest.mark.parametrize(
    ("probs", "payoffs"),
    [
        ([0.5, 0.4], [1.0, 2.0]),  # does not add up to 1
        ([1.2, -0.2], [1.0, 2.0]),  # outside [0, 1]
        ([], []),  # no outcomes
        (1.0, 5.0),  # scalar, no outcome axis
        ([0.5, 0.5], [1.0, np.nan]),
        ([0.5, 0.5], [1.0, np.inf]),
        ([0.5, 0.5], [1.0, 2.0, 3.0]),  # incompatible shapes
    ],
)
def test_fair_game_invalid_values(probs, payoffs) -> None:
    with pytest.raises(ValueError):
        fair_game_expectation(probs, payoffs)


@pytest.mark.parametrize("bad", ["0.5", None, [True, False], [0.5 + 1j, 0.5]])
def test_fair_game_non_numeric(bad) -> None:
    with pytest.raises(TypeError):
        fair_game_expectation(bad, [1.0, 2.0])
    with pytest.raises(TypeError):
        fair_game_expectation([0.5, 0.5], bad)


def test_fair_game_overflow() -> None:
    # E[Y] <= max(Y), but the probabilities may add up to 1 + 1e-10 (within
    # the tolerance), which is enough to push the largest float to inf.
    big = np.finfo(np.float64).max
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(OverflowError):
            fair_game_expectation([0.5, 0.5 + 1e-10], [big, big])


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"rounds": -1}, ValueError),
        ({"rounds": 2.0}, TypeError),
        ({"rounds": True}, TypeError),
        ({"stake": np.nan}, ValueError),
        ({"stake": "1"}, TypeError),
        ({"stake": [1.0, 2.0, 3.0]}, ValueError),  # vs. 2 games
    ],
)
def test_unfair_game_invalid_inputs(kwargs, error) -> None:
    args = {"stake": 1.0, **kwargs}
    stake = args.pop("stake")
    with pytest.raises(error):
        unfair_game_expectation(
            [[0.5, 0.5], [0.5, 0.5]], [0.0, 2.0], stake, **args
        )


def test_unfair_game_overflow() -> None:
    big = np.finfo(np.float64).max
    with pytest.raises(OverflowError):
        unfair_game_expectation([1.0], [big], -big)


# Gambler's ruin: exact formula.


@pytest.mark.parametrize("p", [0.1, 0.4, 0.5, 0.55, 0.9])
@pytest.mark.parametrize("z", [1, 4, 9])
def test_ruin_matches_linear_system(z: int, p: float) -> None:
    expected = ruin_by_linear_system(z, 10, p)
    np.testing.assert_allclose(
        gambler_ruin_probability(z, 10, p), expected, rtol=1e-10, atol=1e-14
    )


def test_ruin_known_values() -> None:
    assert gambler_ruin_probability(3, 10, 0.5).item() == pytest.approx(0.7)
    # r = 1.5: (1.5**5 - 1.5**10) / (1 - 1.5**10)
    expected = (1.5**5 - 1.5**10) / (1 - 1.5**10)
    assert gambler_ruin_probability(5, 10, 0.4).item() == pytest.approx(
        expected, rel=1e-12
    )


def test_ruin_boundaries() -> None:
    np.testing.assert_array_equal(
        gambler_ruin_probability([0, 10], 10, 0.3), [1.0, 0.0]
    )
    np.testing.assert_array_equal(
        gambler_ruin_probability(5, 10, [0.0, 1.0]), [1.0, 0.0]
    )
    assert gambler_ruin_probability(0, 0, 0.5).item() == 1.0
    assert gambler_ruin_probability(4, np.inf, 1.0).item() == 0.0


def test_symmetry_between_players() -> None:
    # The gambler's ruin is the opponent's win: P_z(p) = 1 - P_{a-z}(1 - p).
    z = np.arange(0, 11)
    for p in (0.3, 0.5, 0.62):
        np.testing.assert_allclose(
            gambler_ruin_probability(z, 10, p),
            1 - gambler_ruin_probability(10 - z, 10, 1 - p),
            atol=1e-14,
        )


def test_fair_game_ruin_keeps_expected_fortune() -> None:
    # In a fair game the expected final fortune equals the initial one:
    # 0 * P + a * (1 - P) = z.
    z, a = 3, 10
    ruin = gambler_ruin_probability(z, a, 0.5)
    final = fair_game_expectation(np.stack([ruin, 1 - ruin]), [0.0, a])
    assert final.item() == pytest.approx(z)


def test_ruin_is_monotone() -> None:
    p = np.linspace(0.05, 0.95, 19)
    assert np.all(np.diff(gambler_ruin_probability(5, 10, p)) < 0)
    z = np.arange(0, 21)
    assert np.all(np.diff(gambler_ruin_probability(z, 20, 0.47)) < 0)


def test_infinite_target() -> None:
    np.testing.assert_allclose(
        gambler_ruin_probability(5, np.inf, 0.6), (0.4 / 0.6) ** 5
    )
    np.testing.assert_array_equal(
        gambler_ruin_probability(5, np.inf, [0.3, 0.5]), [1.0, 1.0]
    )
    # A large finite target approaches the infinite one.
    np.testing.assert_allclose(
        gambler_ruin_probability(5, 2_000, 0.6),
        gambler_ruin_probability(5, np.inf, 0.6),
        rtol=1e-12,
    )


def test_relation_with_discrete_time_ruin_model() -> None:
    # Kaas et al. (2008), Section 4.5: U(n) = u + G_1 + ... + G_n with
    # G = +1 (prob. p) or -1 (prob. q), ruin when U < 0. The adjustment
    # coefficient solves m_G(-R) = 1, which gives R = log(p / q). Ruin at
    # U < 0 from u is the gambler's ruin from z = u + 1, and since U(T) = -1
    # at ruin, eq. (4.36) gives exactly exp(-R u) / exp(R).
    p = 0.6
    q = 1 - p
    adj = np.log(p / q)
    assert p * np.exp(-adj) + q * np.exp(adj) == pytest.approx(1.0)
    u = np.arange(0, 15)
    psi = gambler_ruin_probability(u + 1, np.inf, p)
    np.testing.assert_allclose(psi, np.exp(-adj * u) / np.exp(adj))
    assert np.all(psi <= np.exp(-adj * u))  # bound (4.37)


def test_ruin_is_stable_near_a_fair_game() -> None:
    for eps in (1e-6, 1e-10, 1e-14):
        for p in (0.5 + eps, 0.5 - eps):
            assert gambler_ruin_probability(3, 10, p).item() == pytest.approx(
                0.7, abs=1e-5
            )


def ruin_exact(z: int, a: int, p: float) -> float:
    """Ruin probability in exact rational arithmetic, rounded once at the end.

    ``p`` is converted to the exact binary fraction it stores, so the only
    rounding is the final conversion to float.
    """
    p_ = fractions.Fraction(p)
    q_ = 1 - p_
    if p_ == q_:
        return float(fractions.Fraction(a - z, a))
    r = q_ / p_
    return float((r**z - r**a) / (1 - r**a))


@pytest.mark.parametrize(
    ("p", "z", "a", "expected"),
    [
        # Reported in the review of PR #3: the old formula returned 0.0.
        (0.6, 100, 1000, 2.4596544265798527e-18),
        (0.9, 30, 1000, 2.3589824875925706e-29),
    ],
)
def test_small_ruin_probabilities_regression(p, z, a, expected) -> None:
    # atol=0 so that 0.0 cannot pass as an approximation of 1e-18.
    result = gambler_ruin_probability(z, a, p)
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=0)
    np.testing.assert_allclose(result, ruin_exact(z, a, p), rtol=1e-12, atol=0)


@pytest.mark.parametrize("p", [0.05, 0.3, 0.49, 0.5 - 1e-9, 0.5, 0.51, 0.75])
@pytest.mark.parametrize(("z", "a"), [(1, 2), (3, 10), (50, 100), (99, 100)])
def test_ruin_has_full_relative_precision(p, z, a) -> None:
    # Relative error against exact arithmetic, also when P is tiny or close
    # to 1, so no branch loses digits by cancellation.
    result = gambler_ruin_probability(z, a, p).item()
    assert result == pytest.approx(ruin_exact(z, a, p), rel=1e-12, abs=0)


def test_tiny_ruin_probabilities_keep_relative_precision() -> None:
    for p, z, a in [(0.6, 300, 2000), (0.99, 10, 50), (0.75, 600, 700)]:
        expected = ruin_exact(z, a, p)
        # Below machine epsilon: the old formula rounded these to 0.
        assert 0 < expected < np.finfo(np.float64).eps
        np.testing.assert_allclose(
            gambler_ruin_probability(z, a, p), expected, rtol=1e-11, atol=0
        )


def test_fair_game_close_to_target_is_exact() -> None:
    a = 10**12
    assert gambler_ruin_probability(a - 1, a, 0.5).item() == 1 / a


def test_ruin_does_not_overflow_for_large_targets() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        unfavourable = gambler_ruin_probability(10, 1_000_000, 0.4)
        favourable = gambler_ruin_probability(10, 1_000_000, 0.6)
    assert unfavourable.item() == 1.0
    assert favourable.item() == pytest.approx((0.4 / 0.6) ** 10, rel=1e-12)


def test_ruin_scalar_and_broadcasting() -> None:
    scalar = gambler_ruin_probability(3, 10, 0.5)
    assert isinstance(scalar, np.ndarray)
    assert scalar.ndim == 0
    grid = gambler_ruin_probability([[1], [5]], 10, [0.4, 0.5, 0.6])
    assert grid.shape == (2, 3)
    assert gambler_ruin_probability([], 10, 0.5).shape == (0,)


@pytest.mark.parametrize(
    ("z", "a", "p"),
    [
        (2.5, 10, 0.5),  # not a whole number of stakes
        (3, 10.5, 0.5),
        (11, 10, 0.5),  # z > a
        (-1, 10, 0.5),
        (3, -np.inf, 0.5),
        (3, 10, 1.5),
        (3, 10, np.nan),
        (np.inf, np.inf, 0.5),
        ([1, 2], 10, [0.1, 0.2, 0.3]),  # incompatible shapes
    ],
)
def test_ruin_invalid_values(z, a, p) -> None:
    with pytest.raises(ValueError):
        gambler_ruin_probability(z, a, p)


@pytest.mark.parametrize("bad", ["3", None, True, 3 + 0j])
def test_ruin_non_numeric(bad) -> None:
    with pytest.raises(TypeError):
        gambler_ruin_probability(bad, 10, 0.5)
    with pytest.raises(TypeError):
        gambler_ruin_probability(3, bad, 0.5)
    with pytest.raises(TypeError):
        gambler_ruin_probability(3, 10, bad)


# Gambler's ruin: simulation.


@pytest.mark.parametrize(("z", "a", "p"), [(5, 10, 0.5), (5, 10, 0.45)])
def test_simulation_matches_formula(z: int, a: int, p: float) -> None:
    sim = simulate_gambler_ruin(z, a, p, n_paths=40_000, rng=20261017)
    exact = gambler_ruin_probability(z, a, p).item()
    assert sim.unfinished_paths == 0
    assert abs(sim.ruin_probability - exact) < 5 * sim.standard_error
    # Expected duration: within 3 % (the standard error is about 0.5 %).
    assert sim.mean_duration == pytest.approx(
        expected_duration(z, a, p), rel=0.03
    )


def test_simulation_with_infinite_target() -> None:
    sim = simulate_gambler_ruin(
        3, np.inf, 0.6, n_paths=20_000, max_rounds=5_000, rng=7
    )
    exact = gambler_ruin_probability(3, np.inf, 0.6).item()  # 0.296
    assert abs(sim.ruin_probability - exact) < 5 * sim.standard_error + 1e-3
    assert sim.unfinished_paths > 0  # survivors keep playing


def test_simulation_edge_cases() -> None:
    ruined = simulate_gambler_ruin(0, 10, 0.5, n_paths=10, rng=1)
    assert ruined.ruin_probability == 1.0
    assert ruined.mean_duration == 0.0
    at_target = simulate_gambler_ruin(10, 10, 0.5, n_paths=10, rng=1)
    assert at_target.ruin_probability == 0.0
    never_ends = simulate_gambler_ruin(
        3, np.inf, 1.0, n_paths=10, max_rounds=50, rng=1
    )
    assert never_ends.unfinished_paths == 10
    assert np.isnan(never_ends.mean_duration)


def test_simulation_is_reproducible() -> None:
    first = simulate_gambler_ruin(4, 9, 0.48, n_paths=2_000, rng=3)
    second = simulate_gambler_ruin(
        4, 9, 0.48, n_paths=2_000, rng=np.random.default_rng(3)
    )
    assert first == second


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"n_paths": 0}, ValueError),
        ({"max_rounds": 0}, ValueError),
        ({"n_paths": 10.0}, TypeError),
        ({"z": [1, 2]}, ValueError),
        ({"z": 11}, ValueError),
        ({"p": 2.0}, ValueError),
    ],
)
def test_simulation_invalid_inputs(kwargs, error) -> None:
    args = {"z": 3, "a": 10, "p": 0.5, **kwargs}
    z, a, p = args.pop("z"), args.pop("a"), args.pop("p")
    with pytest.raises(error):
        simulate_gambler_ruin(z, a, p, **args)


def test_simulation_result_is_frozen() -> None:
    sim = simulate_gambler_ruin(3, 10, 0.5, n_paths=100, rng=1)
    with pytest.raises(AttributeError):
        sim.ruin_probability = 0.0

import doctest
import warnings

import numpy as np
import pytest

from therisktheory import credibility
from therisktheory.credibility import premium as premium_module

# Kaas et al. (2008), Example 8.2.1, p. 205: 3 groups observed for 5 years.
KAAS_821 = np.array(
    [
        [99.3, 93.7, 103.9, 92.5, 110.6],
        [112.5, 108.3, 118.0, 99.4, 111.8],
        [129.2, 140.9, 108.3, 105.0, 116.6],
    ]
)


def test_docstring_examples() -> None:
    result = doctest.testmod(premium_module)
    assert result.attempted > 0
    assert result.failed == 0


def test_kaas_example_8_2_5() -> None:
    # Group means 100, 110, 120; overall mean 110; F = MSB / MSW = 4.588 and
    # z = 1 - 1/F = 0.782 (Example 8.2.5, p. 210).
    n_groups, n_years = KAAS_821.shape
    group_means = KAAS_821.mean(axis=1)
    overall = KAAS_821.mean()
    msb = n_years * np.sum((group_means - overall) ** 2) / (n_groups - 1)
    msw = np.sum((KAAS_821 - group_means[:, None]) ** 2) / (
        n_groups * (n_years - 1)
    )
    np.testing.assert_allclose(group_means, [100.0, 110.0, 120.0])
    assert msb / msw == pytest.approx(4.588, abs=5e-4)
    z = 1 - msw / msb
    assert z == pytest.approx(0.782, abs=5e-4)

    result = credibility.credibility_premium(group_means, overall, round(z, 3))
    # The book prints 118.82 for the third group, but 0.782 * 120 +
    # 0.218 * 110 = 117.82, and the premiums must be symmetric around 110.
    np.testing.assert_allclose(result, [102.18, 110.0, 117.82], atol=1e-10)


def test_extremes_of_the_credibility_factor() -> None:
    x = np.array([90.0, 135.5])
    np.testing.assert_array_equal(
        credibility.credibility_premium(x, 110.0, 0.0), 110.0
    )
    np.testing.assert_array_equal(
        credibility.credibility_premium(x, 110.0, 1.0), x
    )


def test_premium_moves_monotonically_with_z() -> None:
    z = np.linspace(0.0, 1.0, 21)
    above = credibility.credibility_premium(130.0, 100.0, z)
    below = credibility.credibility_premium(70.0, 100.0, z)
    assert np.all(np.diff(above) > 0)
    assert np.all(np.diff(below) < 0)


def test_components_add_up() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(100, 30, size=50)
    m = rng.normal(100, 5, size=50)
    z = rng.uniform(size=50)
    c = credibility.credibility_components(x, m, z)
    np.testing.assert_allclose(
        c["collective_premium"] + c["experience_correction"], c["premium"]
    )
    np.testing.assert_allclose(
        c["experience_correction"] + c["unexplained_deviation"], x - m
    )
    np.testing.assert_array_equal(c["credibility_factor"], z)


def test_virtual_experience_remark_8_2_7() -> None:
    # (s2 m + a T X_j) / (s2 + a T) = (m k + T X_j) / (k + T), k = s2 / a,
    # equals the credibility premium with z = a T / (a T + s2).
    a, s2, periods, m = 100.0, 64.0, 5, 100.0
    x = np.array([92.0, 101.0, 117.0])
    k = s2 / a
    virtual = (m * k + periods * x) / (k + periods)
    z = a * periods / (a * periods + s2)
    np.testing.assert_allclose(
        credibility.credibility_premium(x, m, z), virtual
    )


def test_poisson_gamma_rate_is_a_credibility_premium() -> None:
    # Kaas eq. (8.60), p. 225: (alpha + x_sum) / (tau + T) is the credibility
    # premium with z = T / (tau + T), individual mean x_sum / T and collective
    # mean alpha / tau. This is the Pólya intensity of Topic 3.
    alpha, tau, years = 2.0, 4.0, 4.0
    x_sum = np.array([0.0, 1.0, 3.0, 7.0])
    expected = (alpha + x_sum) / (tau + years)
    np.testing.assert_allclose(
        credibility.credibility_premium(
            x_sum / years, alpha / tau, years / (tau + years)
        ),
        expected,
    )


def test_buhlmann_premium_is_unbiased_and_mse_optimal() -> None:
    # Theorem 8.2.4: in the balanced Bühlmann model the inhomogeneous linear
    # predictor of X_{j,T+1} with minimal MSE is z X_j + (1 - z) m with
    # z = a T / (a T + s2). Analytically, MSE(z) = (a + s2) - 2 z a
    # + z**2 (a + s2 / T), minimal at that z. Checked by simulation.
    m, a, s2, periods, n_risks = 100.0, 100.0, 64.0, 5, 400_000
    rng = np.random.default_rng(20261107)
    risk = rng.normal(0.0, np.sqrt(a), size=n_risks)
    years = (
        m
        + risk[:, None]
        + rng.normal(0.0, np.sqrt(s2), size=(n_risks, periods + 1))
    )
    x_bar = years[:, :periods].mean(axis=1)
    next_year = years[:, periods]

    z_opt = a * periods / (a * periods + s2)
    grid = np.array([0.0, 0.5, z_opt, 0.95, 1.0])
    mse = [
        np.mean((next_year - credibility.credibility_premium(x_bar, m, z)) ** 2)
        for z in grid
    ]
    analytic = (a + s2) - 2 * grid * a + grid**2 * (a + s2 / periods)
    np.testing.assert_allclose(mse, analytic, rtol=0.01)
    assert np.argmin(mse) == 2  # the Bühlmann factor wins

    bias = np.mean(credibility.credibility_premium(x_bar, m, z_opt) - next_year)
    assert abs(bias) < 5 * np.sqrt(analytic[2] / n_risks)


def test_premiums_stay_between_extremes() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1e6, size=1000)
    m = rng.normal(0, 1e6, size=1000)
    z = rng.uniform(size=1000)
    props = credibility.credibility_premium_properties(x, m, z)
    assert props["within_bounds"].all()
    np.testing.assert_array_equal(props["lower_bound"], np.minimum(x, m))


def test_balanced_portfolio_is_in_balance() -> None:
    group_means = KAAS_821.mean(axis=1)
    props = credibility.credibility_premium_properties(
        group_means, KAAS_821.mean(), 0.782
    )
    assert props["portfolio_imbalance"].item() == pytest.approx(0.0, abs=1e-10)


def test_buhlmann_straub_balance_with_credibility_weighted_mean() -> None:
    # With z_j = a w_j / (s2 + a w_j), the premiums reproduce the total
    # exposure-weighted claims when m is the credibility weighted mean X_zw
    # of (8.32), but in general not with the ordinary weighted mean X_ww.
    rng = np.random.default_rng(7)
    a, s2 = 100.0, 64.0
    w = rng.uniform(0.5, 20.0, size=12)
    x = rng.normal(100.0, 12.0, size=12)
    z = a * w / (s2 + a * w)
    x_zw = np.sum(z * x) / np.sum(z)
    x_ww = np.sum(w * x) / np.sum(w)
    balanced = credibility.credibility_premium_properties(
        x, x_zw, z, exposures=w
    )
    assert balanced["portfolio_imbalance"].item() == pytest.approx(
        0.0, abs=1e-9
    )
    ordinary = credibility.credibility_premium_properties(
        x, x_ww, z, exposures=w
    )
    assert abs(ordinary["portfolio_imbalance"].item()) > 1.0


def test_properties_along_an_axis() -> None:
    # Two portfolios of three risks each, one collective mean per portfolio.
    x = np.array([[100.0, 110.0, 120.0], [50.0, 60.0, 70.0]])
    m = np.array([[110.0], [60.0]])
    props = credibility.credibility_premium_properties(x, m, 0.5)
    assert props["portfolio_imbalance"].shape == (2,)
    np.testing.assert_allclose(
        props["portfolio_imbalance"], [0.0, 0.0], atol=1e-12
    )
    by_column = credibility.credibility_premium_properties(
        x.T, m.T, 0.5, axis=0
    )
    np.testing.assert_allclose(
        by_column["portfolio_imbalance"], [0.0, 0.0], atol=1e-12
    )


def test_broadcasting_scalars_and_empty() -> None:
    grid = credibility.credibility_premium(
        [[100.0], [120.0]], 110.0, [0.0, 0.5, 1.0]
    )
    assert grid.shape == (2, 3)
    scalar = credibility.credibility_premium(1.0, 2.0, 0.5)
    assert isinstance(scalar, np.ndarray)
    assert scalar.ndim == 0
    assert credibility.credibility_premium([], 1.0, 0.5).shape == (0,)
    assert credibility.credibility_components([], 1.0, 0.5)[
        "premium"
    ].shape == (0,)


def test_negative_values_are_allowed() -> None:
    # Loss ratios minus a target, or results, can be negative.
    assert credibility.credibility_premium(-10.0, 10.0, 0.25).item() == 5.0


@pytest.mark.parametrize(
    ("args", "kwargs", "error"),
    [
        ((100.0, 110.0, 1.5), {}, ValueError),
        ((100.0, 110.0, -0.1), {}, ValueError),
        ((100.0, 110.0, np.nan), {}, ValueError),
        ((np.inf, 110.0, 0.5), {}, ValueError),
        ((100.0, np.nan, 0.5), {}, ValueError),
        (("100", 110.0, 0.5), {}, TypeError),
        ((100.0, None, 0.5), {}, TypeError),
        ((100.0, 110.0, True), {}, TypeError),
        (([1.0, 2.0], 110.0, [0.1, 0.2, 0.3]), {}, ValueError),
    ],
)
def test_invalid_inputs(args, kwargs, error) -> None:
    for function in (
        credibility.credibility_premium,
        credibility.credibility_components,
    ):
        with pytest.raises(error):
            function(*args, **kwargs)


@pytest.mark.parametrize(
    ("args", "kwargs", "error"),
    [
        ((100.0, 110.0, 0.5), {}, ValueError),  # not a portfolio
        (([100.0, 120.0], 110.0, 0.5), {"exposures": [-1.0, 1.0]}, ValueError),
        (
            ([100.0, 120.0], 110.0, 0.5),
            {"exposures": [1.0, 2.0, 3.0]},
            ValueError,
        ),
        (([100.0, 120.0], 110.0, 0.5), {"exposures": "a"}, TypeError),
    ],
)
def test_invalid_property_inputs(args, kwargs, error) -> None:
    with pytest.raises(error):
        credibility.credibility_premium_properties(*args, **kwargs)


def test_overflow() -> None:
    big = np.finfo(np.float64).max
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # True components 1.5 * max and 0.5 * (2 max) = max: the first
        # one really overflows.
        with pytest.raises(OverflowError):
            credibility.credibility_components(big, -big, 0.75)
        with pytest.raises(OverflowError):
            credibility.credibility_premium_properties(
                [big, -big], 0.0, 0.0, exposures=[2.0, 2.0]
            )


@pytest.mark.parametrize(
    ("x", "m", "z", "correction", "unexplained", "premium"),
    [
        # Reported in the review of PR #18: x - m overflows, the results
        # do not.
        (1e308, -1e308, 0.5, 1e308, 1e308, 0.0),
        (-1e308, 1e308, 0.5, -1e308, -1e308, 0.0),
        (1e308, -1e308, 0.25, 5e307, 1.5e308, -5e307),
        (
            np.finfo(np.float64).max,
            -np.finfo(np.float64).max,
            0.5,
            np.finfo(np.float64).max,
            np.finfo(np.float64).max,
            0.0,
        ),
    ],
)
def test_components_no_spurious_overflow(
    x, m, z, correction, unexplained, premium
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        parts = credibility.credibility_components(x, m, z)
    assert parts["experience_correction"] == pytest.approx(
        correction, rel=1e-15
    )
    assert parts["unexplained_deviation"] == pytest.approx(
        unexplained, rel=1e-15
    )
    assert parts["premium"] == pytest.approx(premium, abs=1e292)


def test_components_keep_accuracy_when_close() -> None:
    # Close values: x - m is exact here, so z (x - m) must be used
    # rather than z x - z m, which would lose the small difference.
    x = np.array([1.0 + 2.0**-40, 1e16 + 2.0, 110.0])
    m = np.array([1.0, 1e16, 110.0])
    z = 1.0 / 3.0
    parts = credibility.credibility_components(x, m, z)
    np.testing.assert_array_equal(parts["experience_correction"], z * (x - m))
    np.testing.assert_array_equal(
        parts["unexplained_deviation"], (1.0 - z) * (x - m)
    )
    np.testing.assert_array_equal(
        parts["experience_correction"] + parts["unexplained_deviation"],
        [2.0**-40, 2.0, 0.0],
    )


def test_balance_with_full_credibility_for_any_mean() -> None:
    # sum c_j (1 - z_j) = 0: the portfolio is balanced for any m.
    for m in (-50.0, 0.0, 110.0, 1e6):
        props = credibility.credibility_premium_properties(
            [100.0, 120.0, 90.0], m, [1.0, 1.0, 0.3], exposures=[2.0, 5.0, 0.0]
        )
        assert float(props["portfolio_imbalance"]) == 0.0

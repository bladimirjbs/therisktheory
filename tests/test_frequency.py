import doctest
import math
import warnings

import numpy as np
import pytest
from scipy import integrate, stats

from therisktheory import frequency
from therisktheory.frequency import counts, processes
from therisktheory.solvency_pricing import premiums

N_GRID = np.arange(0, 400)


def nb_pmf_direct(n: int, r: float, p: float) -> float:
    """C(r + n - 1, n) p**r (1 - p)**n with gamma functions (Kaas, 3.74)."""
    log_c = math.lgamma(r + n) - math.lgamma(n + 1) - math.lgamma(r)
    return math.exp(log_c + r * math.log(p) + n * math.log1p(-p))


def moments_from_pmf(pmf: np.ndarray) -> tuple[float, float, float]:
    """Mean, variance and third central moment by summing the pmf."""
    mean = float(np.sum(N_GRID * pmf))
    var = float(np.sum((N_GRID - mean) ** 2 * pmf))
    third = float(np.sum((N_GRID - mean) ** 3 * pmf))
    return mean, var, third


@pytest.mark.parametrize("module", [counts, processes])
def test_docstring_examples(module) -> None:
    result = doctest.testmod(module)
    assert result.attempted > 0
    assert result.failed == 0


def test_scipy_is_imported_lazily() -> None:
    import subprocess
    import sys

    code = (
        "import sys, therisktheory.frequency; "
        "print('scipy.stats' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"


# Poisson.


@pytest.mark.parametrize("lam", [0.0, 0.3, 2.0, 15.0])
def test_poisson_pmf_matches_formula(lam: float) -> None:
    n = np.arange(0, 40)
    direct = [math.exp(-lam) * lam**k / math.factorial(k) for k in n]
    np.testing.assert_allclose(
        frequency.poisson_count_pmf(n, rate=lam), direct, rtol=1e-12, atol=0
    )


def test_poisson_pmf_negative_n_is_zero() -> None:
    assert frequency.poisson_count_pmf(-1, rate=2.0).item() == 0.0


@pytest.mark.parametrize("lam", [0.5, 3.0, 20.0])
def test_poisson_moments_match_pmf_sums(lam: float) -> None:
    pmf = frequency.poisson_count_pmf(N_GRID, rate=lam)
    assert np.sum(pmf) == pytest.approx(1.0, abs=1e-12)
    mean, var, third = moments_from_pmf(pmf)
    m = frequency.poisson_count_moments(rate=lam)
    assert m["mean"].item() == pytest.approx(mean, rel=1e-10)
    assert m["variance"].item() == pytest.approx(var, rel=1e-10)
    assert m["third_central_moment"].item() == pytest.approx(third, rel=1e-8)
    assert m["second_raw_moment"].item() == pytest.approx(var + mean**2)


def test_poisson_cdf_is_cumulative_pmf() -> None:
    n = np.arange(0, 30)
    np.testing.assert_allclose(
        frequency.poisson_count_cdf(n, rate=4.0),
        np.cumsum(frequency.poisson_count_pmf(n, rate=4.0)),
        rtol=1e-12,
    )
    # Step function between integers, 0 below the support, 1 at +inf.
    np.testing.assert_allclose(
        frequency.poisson_count_cdf([-0.5, 2.0, 2.99, np.inf], rate=4.0),
        [0.0, *frequency.poisson_count_cdf([2, 2], rate=4.0), 1.0],
    )


@pytest.mark.parametrize("lam", [0.0, 0.7, 4.0])
def test_poisson_ppf_inverts_cdf(lam: float) -> None:
    q = np.linspace(0.0, 0.999, 50)
    n = frequency.poisson_count_ppf(q, rate=lam)
    assert np.all(frequency.poisson_count_cdf(n, rate=lam) >= q - 1e-12)
    below = frequency.poisson_count_cdf(n - 1, rate=lam)
    assert np.all((below < q) | (n == 0))


def test_poisson_ppf_at_one() -> None:
    assert frequency.poisson_count_ppf(1.0, rate=2.0).item() == np.inf


# Negative binomial.


@pytest.mark.parametrize(("r", "p"), [(1.0, 0.3), (2.5, 0.6), (0.4, 0.9)])
def test_negative_binomial_pmf_matches_formula(r: float, p: float) -> None:
    n = np.arange(0, 40)
    direct = [nb_pmf_direct(int(k), r, p) for k in n]
    np.testing.assert_allclose(
        frequency.negative_binomial_count_pmf(n, size=r, probability=p),
        direct,
        rtol=1e-12,
        atol=0,
    )


def test_geometric_is_negative_binomial_with_size_one() -> None:
    # Table A: geometric(p) = negative binomial(1, p).
    n = np.arange(0, 20)
    np.testing.assert_allclose(
        frequency.negative_binomial_count_pmf(n, size=1.0, probability=0.25),
        0.25 * 0.75**n,
        rtol=1e-13,
    )


@pytest.mark.parametrize(("r", "p"), [(1.0, 0.3), (2.5, 0.6), (8.0, 0.8)])
def test_negative_binomial_moments_match_pmf_sums(r: float, p: float) -> None:
    pmf = frequency.negative_binomial_count_pmf(N_GRID, size=r, probability=p)
    assert np.sum(pmf) == pytest.approx(1.0, abs=1e-10)
    mean, var, third = moments_from_pmf(pmf)
    m = frequency.negative_binomial_count_moments(size=r, probability=p)
    assert m["mean"].item() == pytest.approx(mean, rel=1e-9)
    assert m["variance"].item() == pytest.approx(var, rel=1e-9)
    assert m["third_central_moment"].item() == pytest.approx(third, rel=1e-7)


def test_negative_binomial_overdispersion() -> None:
    # Var[N] / E[N] = 1 / p >= 1 (Section 3.9.4).
    p = np.array([0.2, 0.5, 0.9, 1.0])
    m = frequency.negative_binomial_count_moments(size=3.0, probability=p)
    with np.errstate(invalid="ignore"):
        ratio = m["variance"] / m["mean"]
    np.testing.assert_allclose(ratio[:3], 1 / p[:3])
    assert m["mean"][3] == 0.0  # p = 1: point mass at 0


def test_negative_binomial_cdf_and_ppf() -> None:
    n = np.arange(0, 30)
    cdf = frequency.negative_binomial_count_cdf(n, size=2.0, probability=0.4)
    np.testing.assert_allclose(
        cdf,
        np.cumsum(
            frequency.negative_binomial_count_pmf(n, size=2.0, probability=0.4)
        ),
        rtol=1e-12,
    )
    q = np.linspace(0.0, 0.99, 40)
    k = frequency.negative_binomial_count_ppf(q, size=2.0, probability=0.4)
    assert np.all(
        frequency.negative_binomial_count_cdf(k, size=2.0, probability=0.4)
        >= q - 1e-12
    )
    assert k[0] == 0.0


def test_mixed_poisson_with_gamma_structure_is_negative_binomial() -> None:
    # Kaas et al. (2008), Example 3.3.1: N | Lambda ~ Poisson(Lambda) and
    # Lambda ~ gamma(alpha, beta) give negative binomial(alpha, beta/(beta+1)).
    alpha, beta = 2.5, 0.8
    for n in range(0, 12):
        mixed, _ = integrate.quad(
            lambda lam, n=n: (
                stats.poisson.pmf(n, lam)
                * stats.gamma.pdf(lam, alpha, scale=1 / beta)
            ),
            0,
            np.inf,
        )
        nb = frequency.negative_binomial_count_pmf(
            n, size=alpha, probability=beta / (beta + 1)
        )
        assert nb.item() == pytest.approx(mixed, rel=1e-8)


# Probability generating functions.


def test_pgf_at_one_and_series() -> None:
    s = np.array([-0.9, 0.0, 0.5, 1.0])
    for pgf, pmf in [
        (
            lambda x: frequency.poisson_count_pgf(x, rate=2.0),
            frequency.poisson_count_pmf(N_GRID, rate=2.0),
        ),
        (
            lambda x: frequency.negative_binomial_count_pgf(
                x, size=2.0, probability=0.4
            ),
            frequency.negative_binomial_count_pmf(
                N_GRID, size=2.0, probability=0.4
            ),
        ),
    ]:
        series = [np.sum(pmf * x**N_GRID) for x in s]
        np.testing.assert_allclose(pgf(s), series, rtol=1e-10, atol=1e-14)
        assert pgf(1.0).item() == pytest.approx(1.0)


def test_pgf_accepts_complex_arguments() -> None:
    # Characteristic function of N: phi_N(u) = P_N(exp(i u)).
    u = np.array([0.3, 1.0, 2.0])
    z = np.exp(1j * u)
    phi_poisson = frequency.poisson_count_pgf(z, rate=3.0)
    assert phi_poisson.dtype == np.complex128
    np.testing.assert_allclose(phi_poisson, np.exp(3.0 * (z - 1)))
    pmf = frequency.negative_binomial_count_pmf(
        N_GRID, size=2.0, probability=0.4
    )
    series = [np.sum(pmf * zz**N_GRID) for zz in z]
    np.testing.assert_allclose(
        frequency.negative_binomial_count_pgf(z, size=2.0, probability=0.4),
        series,
        rtol=1e-10,
    )


def test_pgf_derivative_at_one_is_the_mean() -> None:
    # Complex-step derivative: P'(1) = Im P(1 + i h) / h = E[N].
    h = 1e-20
    poisson = frequency.poisson_count_pgf(1 + 1j * h, rate=3.0)
    nb = frequency.negative_binomial_count_pgf(
        1 + 1j * h, size=2.0, probability=0.4
    )
    assert poisson.imag / h == pytest.approx(3.0, rel=1e-14)
    assert nb.imag / h == pytest.approx(2.0 * 0.6 / 0.4, rel=1e-14)


def test_compound_mgf_through_pgf() -> None:
    # m_S(t) = P_N(m_X(t)) (Kaas, eq. (3.5)) gives the compound Poisson mgf
    # exp(lambda (m_X(t) - 1)); with exponential(mean 200) claims, small t.
    t, lam, mean_x = 0.001, 3.0, 200.0
    m_x = 1 / (1 - mean_x * t)
    m_s = frequency.poisson_count_pgf(m_x, rate=lam)
    assert m_s.item() == pytest.approx(np.exp(lam * (m_x - 1)))


def test_negative_binomial_pgf_divergence() -> None:
    with pytest.raises(ValueError, match="diverges"):
        frequency.negative_binomial_count_pgf(2.5, size=1.0, probability=0.5)


def test_pgf_overflow() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(OverflowError):
            frequency.poisson_count_pgf(1e10, rate=1e300)


# Claim intensity and Pólya process.


def test_constant_risk_intensity_ignores_history() -> None:
    result = frequency.claim_intensity([0.0, 1.0, 5.0], [0, 3, 10], rate=0.4)
    np.testing.assert_array_equal(result, [0.4, 0.4, 0.4])


def test_polya_intensity_is_posterior_mean() -> None:
    # Kaas et al. (2008), eq. (8.59): E[Lambda | k claims in (0, t)] with a
    # gamma(h, h / lambda) prior. Checked by numerical Bayes integration.
    lam, h, t, k = 0.5, 2.0, 3.0, 4
    prior = stats.gamma(h, scale=lam / h)

    def weight(x: float) -> float:
        return stats.poisson.pmf(k, x * t) * prior.pdf(x)

    num, _ = integrate.quad(lambda x: x * weight(x), 0, np.inf)
    den, _ = integrate.quad(weight, 0, np.inf)
    assert frequency.claim_intensity(
        t, k, rate=lam, shape=h
    ).item() == pytest.approx(num / den, rel=1e-8)


def test_polya_intensity_contagion() -> None:
    k = np.arange(0, 6)
    rising = frequency.claim_intensity(2.0, k, rate=0.5, shape=1.5)
    assert np.all(np.diff(rising) > 0)  # each claim raises the rate
    t = np.linspace(0.0, 10.0, 6)
    falling = frequency.claim_intensity(t, 0, rate=0.5, shape=1.5)
    assert np.all(np.diff(falling) < 0)  # a claim-free period lowers it
    assert falling[0] == 0.5  # no history: prior mean


def test_polya_tends_to_poisson() -> None:
    n = np.arange(0, 15)
    poisson = frequency.poisson_count_pmf(n, rate=2.0)
    assert np.array_equal(
        frequency.polya_process_count_pmf(n, time=2.0, rate=1.0, shape=np.inf),
        poisson,
    )
    close = frequency.polya_process_count_pmf(n, time=2.0, rate=1.0, shape=1e8)
    np.testing.assert_allclose(close, poisson, rtol=1e-6)
    np.testing.assert_allclose(
        frequency.claim_intensity(4.0, 9, rate=1.0, shape=1e12), 1.0, rtol=1e-9
    )


def test_polya_process_moments() -> None:
    m = frequency.polya_process_moments(time=2.0, rate=1.5, shape=4.0)
    params = frequency.polya_process_parameters(time=2.0, rate=1.5, shape=4.0)
    nb = frequency.negative_binomial_count_moments(**params)
    for key in m:
        np.testing.assert_allclose(m[key], nb[key], rtol=1e-13)
    assert m["mean"].item() == 3.0
    assert m["variance"].item() == pytest.approx(3.0 + 9.0 / 4.0)
    inf = frequency.polya_process_moments(time=2.0, rate=1.5, shape=np.inf)
    assert inf["variance"].item() == inf["mean"].item() == 3.0


def test_polya_sample_matches_negative_binomial() -> None:
    draws = frequency.polya_process_sample(
        time=2.0, rate=1.5, shape=4.0, sample_shape=200_000, rng=3
    )
    m = frequency.polya_process_moments(time=2.0, rate=1.5, shape=4.0)
    se_mean = np.sqrt(m["variance"].item() / draws.size)
    assert abs(draws.mean() - m["mean"].item()) < 5 * se_mean
    assert draws.var() == pytest.approx(m["variance"].item(), rel=0.02)
    params = frequency.polya_process_parameters(time=2.0, rate=1.5, shape=4.0)
    freq = np.bincount(draws, minlength=8)[:8] / draws.size
    np.testing.assert_allclose(
        freq,
        frequency.negative_binomial_count_pmf(np.arange(8), **params),
        atol=5e-3,
    )


# Compound Poisson.


def test_compound_poisson_cumulants_and_topic_1() -> None:
    # Exponential(mean 200) claims: E[X**j] = j! 200**j.
    raw = [math.factorial(j) * 200.0**j for j in (1, 2, 3)]
    kappa = frequency.compound_poisson_cumulants(
        rate=3.0, severity_raw_moments=raw, time=2.0
    )
    np.testing.assert_allclose(kappa, [6 * raw[0], 6 * raw[1], 6 * raw[2]])
    # Integration with Topic 1: kappa_1 = E[S] = E[N] E[X].
    assert kappa[0] == premiums.expected_aggregate_loss(6.0, 200.0).item()


def test_compound_poisson_simulation_matches_cumulants() -> None:
    lam, mean_x, n = 3.0, 200.0, 200_000
    sims = frequency.compound_poisson_sample(
        rate=lam,
        severity_sampler=lambda k, gen: gen.exponential(mean_x, size=k),
        sample_shape=n,
        rng=20261031,
    )
    kappa = frequency.compound_poisson_cumulants(
        rate=lam, severity_raw_moments=[mean_x, 2 * mean_x**2, 6 * mean_x**3]
    )
    se = np.sqrt(kappa[1] / n)
    assert abs(sims.mean() - kappa[0]) < 5 * se
    assert sims.var() == pytest.approx(kappa[1], rel=0.02)
    third = np.mean((sims - sims.mean()) ** 3)
    assert third == pytest.approx(kappa[2], rel=0.1)


def test_compound_poisson_merge_theorem() -> None:
    # Theorem 3.4.1: cumulants of S1 + S2 equal those of the merged process.
    rates = np.array([1.0, 3.0])
    raw = np.array([[100.0, 2e4, 6e6], [50.0, 5e3, 7.5e5]])
    merged = frequency.compound_poisson_merge(rates)
    np.testing.assert_allclose(merged["weights"].sum(), 1.0)
    mixed_raw = merged["weights"] @ raw
    separate = sum(
        frequency.compound_poisson_cumulants(rate=r, severity_raw_moments=m)
        for r, m in zip(rates, raw, strict=True)
    )
    np.testing.assert_allclose(
        frequency.compound_poisson_cumulants(
            rate=merged["rate"], severity_raw_moments=mixed_raw
        ),
        separate,
        rtol=1e-13,
    )


def test_compound_poisson_zero_rate() -> None:
    sims = frequency.compound_poisson_sample(
        rate=0.0,
        severity_sampler=lambda k, gen: gen.exponential(1.0, size=k),
        sample_shape=10,
        rng=1,
    )
    np.testing.assert_array_equal(sims, np.zeros(10))


# Shapes, sampling and invalid inputs.


def test_broadcasting_and_empty_arrays() -> None:
    grid = frequency.poisson_count_pmf([[0], [1], [2]], rate=[0.5, 1.0])
    assert grid.shape == (3, 2)
    assert frequency.poisson_count_pmf([], rate=1.0).shape == (0,)
    assert frequency.negative_binomial_count_pmf(
        np.array([], dtype=int), size=1.0, probability=0.5
    ).shape == (0,)
    scalar = frequency.poisson_count_pmf(1, rate=1.0)
    assert isinstance(scalar, np.ndarray)
    assert scalar.ndim == 0


def test_sampling_reproducible_and_shaped() -> None:
    a = frequency.poisson_count_sample(rate=2.0, sample_shape=(3, 4), rng=5)
    b = frequency.poisson_count_sample(
        rate=2.0, sample_shape=(3, 4), rng=np.random.default_rng(5)
    )
    assert a.shape == (3, 4)
    assert a.dtype == np.int64
    np.testing.assert_array_equal(a, b)
    nb = frequency.negative_binomial_count_sample(
        size=2.0, probability=0.4, sample_shape=100_000, rng=1
    )
    assert nb.mean() == pytest.approx(3.0, rel=0.02)


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (lambda: frequency.poisson_count_pmf(1.5, rate=1.0), ValueError),
        (lambda: frequency.poisson_count_pmf(1, rate=-1.0), ValueError),
        (lambda: frequency.poisson_count_pmf(1, rate=np.nan), ValueError),
        (lambda: frequency.poisson_count_pmf(1, rate=np.inf), ValueError),
        (lambda: frequency.poisson_count_pmf("1", rate=1.0), TypeError),
        (lambda: frequency.poisson_count_pmf(1, rate=None), TypeError),
        (lambda: frequency.poisson_count_cdf(np.nan, rate=1.0), ValueError),
        (lambda: frequency.poisson_count_ppf(1.2, rate=1.0), ValueError),
        (
            lambda: frequency.poisson_count_pmf([0, 1], rate=[1.0, 2.0, 3.0]),
            ValueError,
        ),
        (
            lambda: frequency.negative_binomial_count_pmf(
                1, size=0.0, probability=0.5
            ),
            ValueError,
        ),
        (
            lambda: frequency.negative_binomial_count_pmf(
                1, size=1.0, probability=0.0
            ),
            ValueError,
        ),
        (
            lambda: frequency.negative_binomial_count_pmf(
                1, size=1.0, probability=1.5
            ),
            ValueError,
        ),
        (
            lambda: frequency.negative_binomial_count_moments(
                size=True, probability=0.5
            ),
            TypeError,
        ),
        (lambda: frequency.poisson_count_pgf("a", rate=1.0), TypeError),
        (lambda: frequency.poisson_count_pgf(np.nan, rate=1.0), ValueError),
        (
            lambda: frequency.poisson_count_pgf([[1], [2, 3]], rate=1.0),
            TypeError,
        ),
        (lambda: frequency.claim_intensity(1.0, 1.5, rate=1.0), ValueError),
        (lambda: frequency.claim_intensity(1.0, 1, rate=0.0), ValueError),
        (
            lambda: frequency.claim_intensity(1.0, 1, rate=1.0, shape=0.0),
            ValueError,
        ),
        (lambda: frequency.claim_intensity(-1.0, 1, rate=1.0), ValueError),
        (
            lambda: frequency.polya_process_parameters(
                time=1.0, rate=1.0, shape=np.inf
            ),
            ValueError,
        ),
        (
            lambda: frequency.polya_process_count_pmf(
                1, time=1.0, rate=1.0, shape=[1.0, np.inf]
            ),
            ValueError,
        ),
        (
            lambda: frequency.compound_poisson_cumulants(
                rate=1.0, severity_raw_moments=[]
            ),
            ValueError,
        ),
        (
            lambda: frequency.compound_poisson_cumulants(
                rate=1.0, severity_raw_moments=5.0
            ),
            ValueError,
        ),
        (
            lambda: frequency.compound_poisson_cumulants(
                rate=1.0, severity_raw_moments=[1.0, -2.0]
            ),
            ValueError,
        ),
        (lambda: frequency.compound_poisson_merge([0.0, 0.0]), ValueError),
        (lambda: frequency.compound_poisson_merge(2.0), ValueError),
        (lambda: frequency.compound_poisson_merge([]), ValueError),
        (
            lambda: frequency.poisson_count_sample(
                rate=[1.0, 2.0], sample_shape=3
            ),
            ValueError,
        ),
        (
            lambda: frequency.compound_poisson_sample(
                rate=1.0,
                severity_sampler=lambda k, gen: np.ones(k + 1),
                sample_shape=5,
                rng=1,
            ),
            ValueError,
        ),
        (
            lambda: frequency.compound_poisson_sample(
                rate=1.0,
                severity_sampler=lambda k, gen: np.ones(k),
                sample_shape=0,
            ),
            ValueError,
        ),
        (
            lambda: frequency.compound_poisson_sample(
                rate=1.0,
                severity_sampler=lambda k, gen: np.ones(k),
                sample_shape=True,
            ),
            ValueError,
        ),
    ],
)
def test_invalid_inputs(call, error) -> None:
    with pytest.raises(error):
        call()


def test_moment_overflow() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(OverflowError):
            frequency.negative_binomial_count_moments(
                size=1e300, probability=1e-300
            )
        with pytest.raises(OverflowError):
            frequency.poisson_count_moments(rate=1e200)
        with pytest.raises(OverflowError):
            frequency.compound_poisson_cumulants(
                rate=1e300, severity_raw_moments=[1e300]
            )
        with pytest.raises(OverflowError):
            frequency.claim_intensity(0.0, 1e10, rate=1e300, shape=1.0)

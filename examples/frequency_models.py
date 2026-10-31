import numpy as np

from therisktheory import frequency

# Poisson versus negative binomial with the same mean 2.
n = np.arange(6)
print(frequency.poisson_count_pmf(n, rate=2.0).round(4))
print(
    frequency.negative_binomial_count_pmf(n, size=2.0, probability=0.5).round(4)
)
print(frequency.negative_binomial_count_moments(size=2.0, probability=0.5))

# Pólya process: mean intensity 0.5 claims per year, shape 2.
print(frequency.polya_process_parameters(time=4.0, rate=0.5, shape=2.0))
# A driver with 3 claims in 4 years versus one with none.
print(frequency.claim_intensity(4.0, [0, 3], rate=0.5, shape=2.0))
# [0.25 0.625]

# Compound Poisson: 3 claims per year, exponential claims with mean 200.
kappa = frequency.compound_poisson_cumulants(
    rate=3.0, severity_raw_moments=[200.0, 2 * 200.0**2, 6 * 200.0**3]
)
sims = frequency.compound_poisson_sample(
    rate=3.0,
    severity_sampler=lambda k, gen: gen.exponential(200.0, size=k),
    sample_shape=100_000,
    rng=1,
)
print(kappa[:2], sims.mean().round(1), sims.var().round(-3))

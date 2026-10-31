# Topic 3: claim frequency models

Package: `therisktheory.frequency`. Status: experimental.

Page numbers refer to Kaas, Goovaerts, Dhaene and Denuit (2008), *Modern
Actuarial Risk Theory: Using R*, 2nd edition, Springer.

## Public API

| Function | Object |
|---|---|
| `poisson_count_pmf/cdf/ppf/moments/pgf/sample` | `N ~ Poisson(rate)` |
| `negative_binomial_count_pmf/cdf/ppf/moments/pgf/sample` | `N ~ negative binomial(size, probability)` |
| `claim_intensity` | instantaneous claim rate given the history |
| `polya_process_parameters/count_pmf/moments/sample` | Pólya process `N(t)` |
| `compound_poisson_cumulants/merge/sample` | compound Poisson process `S(t)` |

Conventions follow `severity_aggregate`: parameters after the first argument
are keyword-only, outputs are float64 arrays (0-d for scalars), and
`*_moments` returns a dict with `mean`, `variance` and `second_raw_moment`,
plus `third_central_moment`, which the Normal Power and Edgeworth
approximations need.

## Claim count distributions

**Poisson** (Table A, p. 368): `Pr[N = n] = e^{-λ} λ^n / n!`, with all
cumulants equal to `λ`, so `E[N] = Var[N] = E[(N - E N)^3] = λ`.

**Negative binomial** (Table A; eq. (3.74), p. 64), with `q = 1 - p`:

```text
Pr[N = n] = C(r + n - 1, n) p^r q^n,   n = 0, 1, 2, ...
E[N] = r q / p,   Var[N] = r q / p^2,   E[(N - E N)^3] = r q (2 - p) / p^3.
```

`r > 0` need not be an integer. The overdispersion `Var[N]/E[N] = 1/p` is at
least one (Section 3.9.4, p. 64), which is why it is used when the data have
more variance than a Poisson model allows (Section 3.3, p. 45).

**Probability generating function.** `P_N(s) = E[s^N]` is `exp(λ(s - 1))`
for the Poisson and `(p / (1 - q s))^r` for the negative binomial
(`|q s| < 1`). The aggregate model uses it through
`m_S(t) = P_N(m_X(t))` (eq. (3.5), p. 43). Complex `s` is accepted, so the
characteristic function is `φ_S(u) = P_N(φ_X(u))`.

## Constant risk and the Poisson process

In a Poisson process with intensity `λ` (Definition 4.2.1, p. 89) the
increments `N(t + h) - N(t) ~ Poisson(λh)` are independent and stationary,
and the probability of a claim in `(t, t + dt)` is `λ dt` whatever happened
before (eq. (4.5)). This is the constant-risk case: the instantaneous claim
rate is `λ` for every history.

## Variable basic probabilities: mixed Poisson and the Pólya process

If the intensity is a random structure variable `Λ` and, given `Λ`, the
counts are Poisson, then (Example 3.3.1, eqs. (3.13)-(3.14), pp. 45-46)

```text
E[N] = E[Λ],   Var[N] = E[Λ] + Var[Λ] ≥ E[N].
```

With `Λ ~ gamma(α, β)`, eq. (3.15) gives `N ~ negative binomial(α, β/(β+1))`.

The **Pólya process** is the mixed Poisson process with a gamma structure
distribution. We parameterize it by the mean intensity `λ = E[Λ]` and the
shape `h`, so `Λ ~ gamma(h, h/λ)`:

1. Over `(0, t)`, `Λt ~ gamma(h, h/(λt))`, and eq. (3.15) gives
   `N(t) ~ negative binomial(r = h, p = h / (h + λt))`
   (`polya_process_parameters`).
2. Hence `E[N(t)] = λt` and `Var[N(t)] = λt + (λt)^2 / h`.
3. After `k` claims in `(0, t)` the posterior of `Λ` is
   `gamma(h + k, h/λ + t)`. Its mean is the instantaneous claim rate,
   eq. (8.59), p. 225:

   ```text
   λ(t | k) = (h + k) / (h/λ + t) = λ (h + k) / (h + λt).
   ```

   Each claim raises the rate and a claim-free period lowers it. This is the
   contagion of the Pólya process (`claim_intensity`).
4. As `h → ∞` the gamma distribution concentrates at `λ` and everything
   reduces to the Poisson process; `shape=numpy.inf` is accepted for that
   case.

## Compound Poisson process

`S(t) = X_1 + ... + X_{N(t)}` with `N(t) ~ Poisson(λt)`. Its cumulant
generating function is `λt (m_X(u) - 1)`, so the `j`-th cumulant is
`κ_j = λ t E[X^j]` (`compound_poisson_cumulants`). In particular
`κ_1 = E[S(t)] = E[N(t)] E[X]`, the Topic 1 formula.

Independent compound Poisson portfolios with intensities `λ_i` and claim
cdfs `P_i` add up to a compound Poisson portfolio with `λ = Σ λ_i` and claim
cdf `Σ (λ_i/λ) P_i` (Theorem 3.4.1, eq. (3.20), p. 47;
`compound_poisson_merge`).

`compound_poisson_sample` draws all claim counts first and then all claim
amounts in one call to the severity sampler, so it has no Python loop over
simulations. Valera's Monte Carlo work can use it with any severity from
`severity_aggregate`.

## Independent checks (`tests/test_frequency.py`)

- The pmfs are compared with the direct formulas (`math.factorial`,
  `math.lgamma`).
- The moments are compared with sums over the pmf.
- The pgf is compared with its series, including at complex points; its
  complex-step derivative at 1 equals the mean.
- **Example 3.3.1:** the Poisson-gamma mixture is integrated numerically and
  equals the negative binomial pmf.
- **Eq. (8.59):** the posterior mean of `Λ` is computed by numerical Bayes
  integration and equals `claim_intensity`.
- The Pólya simulation (gamma first, then Poisson) matches the negative
  binomial pmf and moments, and the Pólya process tends to the Poisson
  process as `h → ∞`.
- A compound Poisson simulation matches `κ_1`, `κ_2` and `κ_3`; the merged
  portfolio has the same cumulants as the sum (Theorem 3.4.1); and
  `κ_1 = expected_aggregate_loss` from Topic 1.

## Imports

`numpy` is the only import when the package is loaded. `scipy.stats` is
imported inside the functions that need it, because importing it is much
slower than importing the package; a test checks this.

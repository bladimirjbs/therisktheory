# Topic 4: the credibility premium

Module: `therisktheory.credibility.premium`. Status: experimental.

Page numbers refer to Kaas, Goovaerts, Dhaene and Denuit (2008), *Modern
Actuarial Risk Theory: Using R*, 2nd edition, Springer,
https://doi.org/10.1007/978-3-540-70998-5.

## Foundations and objectives

An insurer has to set next year's premium for a risk `j` (a contract or a
group of contracts) with a few years of its own claims experience, inside a
portfolio with a lot more experience (Section 8.1, p. 203). There are two
extreme answers:

- **Charge everyone the collective mean `m`.** This is right if the portfolio
  is homogeneous, but then the good risks leave and only the bad ones stay.
- **Charge each risk its own average `X_j`.** This is right if the portfolio
  is heterogeneous, but only when the individual experience is large enough.

A real portfolio is neither completely homogeneous nor completely
heterogeneous, so credibility theory takes a weighted average, eq. (8.1):

```text
P_j = z_j X_j + (1 - z_j) m,    0 <= z_j <= 1.
```

`z_j` is the **credibility factor**: how much the experience of risk `j` is
trusted. It should be close to one when that experience is large, when it
varies little, or when the risks of the portfolio differ a lot from each
other. The two classical ways to choose it are:

- **limited fluctuation credibility** (full and partial credibility);
- **greatest accuracy credibility** (Bühlmann), where `z_j` minimizes a mean
  squared error (p. 204).

## Construction (balanced Bühlmann model)

The claims of risk `j` in year `t` are decomposed as `X_jt = m + Ξ_j + Ξ_jt`
(eq. (8.5), p. 206), where:

- `Ξ_j` is the risk quality of contract `j`, with mean 0 and variance `a`
  (heterogeneity between risks);
- `Ξ_jt` is the yearly fluctuation, with mean 0 and variance `s²` (variation
  within a risk).

Among the linear predictors of next year's claims `X_{j,T+1}`, the one with
minimal mean squared error is

```text
z X_j + (1 - z) m,   z = aT / (aT + s²)
```

with `m` known (Theorem 8.2.4, eq. (8.16), p. 209), or with `m` replaced by
the overall mean `X` (Theorem 8.2.2, eqs. (8.9)-(8.12), pp. 207-208). In the
Bühlmann-Straub model, with weights `w_j`, `z_j = a w_j / (s² + a w_j)` and
`m` is estimated by the credibility weighted mean `X_zw` (eq. (8.32) and
Theorem 8.4.1, p. 215).

The factor `z` grows with the experience `T` and with the heterogeneity `a`,
and falls with the within-risk variance `s²` (Remark 8.2.3, p. 209).

## Public API: the common credibility premium

Every method of Topic 4 produces a credibility factor, and they all share
this module:

| Function | Result |
|---|---|
| `credibility_premium(individual_mean, collective_mean, weight)` | `z X_j + (1 - z) m` |
| `credibility_components(...)` | `m`, `X_j`, `z`, the correction `z (X_j - m)`, the rest `(1 - z)(X_j - m)` and the premium |
| `credibility_premium_properties(individual_means, collective_mean, weights, *, exposures=1, axis=-1)` | premiums, bounds, `within_bounds`, `portfolio_imbalance` |

`weight` is an input on purpose. The Bühlmann factor comes from
`credibility_weight` (Garcia, #12), the partial and full credibility factors
from Monticelli's and Passanisi's functions, and the Bayesian and
free-parameter ones from Pérez and Valera. Each of them only has to compute
`z`; the premium formula is written once.

```python
from therisktheory import credibility

credibility.credibility_premium([100.0, 110.0, 120.0], 110.0, 0.782)
# array([102.18, 110.  , 117.82])
```

## Components

```text
P_j = m + z (X_j - m)
```

The premium is the collective premium plus the believed part of the
difference between the individual experience and the collective mean. The
remaining `(1 - z)(X_j - m)` is attributed to chance.

**Virtual experience.** The same premium equals `(k m + T X_j) / (k + T)`
with `k = s²/a`. This is the average claims after adding `k` virtual years of
experience at the collective level (Remark 8.2.7, p. 211). With a Poisson
count and a gamma structure, eq. (8.60) (p. 225) writes the Pólya intensity
of Topic 3 in exactly this form.

## Properties

1. **Between the extremes (shrinkage).** Since `0 <= z <= 1`, each premium
   lies between `m` and `X_j`, so the premiums are closer together than the
   individual means (Example 8.2.5, p. 210).
2. **Limits.** `z = 0` gives `m` and `z = 1` gives `X_j`, exactly.
3. **Unbiasedness and optimality.** In the Bühlmann model, `E[P_j] = m`, and
   the Bühlmann `z` gives the smallest MSE among linear predictors
   (Theorem 8.2.4). The MSE as a function of `z` is
   `(a + s²) - 2 z a + z² (a + s²/T)`, minimal at `z = aT/(aT + s²)`.
4. **Portfolio balance.** With exposures `c_j`,

   ```text
   Σ c_j P_j - Σ c_j X_j = Σ c_j (1 - z_j)(m - X_j),
   ```

   which is zero exactly when `m = Σ c_j (1 - z_j) X_j / Σ c_j (1 - z_j)`,
   provided `Σ c_j (1 - z_j) > 0`. If that sum is zero, every term
   `c_j (1 - z_j)` is zero (since `c_j >= 0` and `0 <= z_j <= 1`), for example
   when all risks with positive exposure have full credibility, and the
   portfolio is balanced for any `m`.

   - In the balanced model (equal `z` and `c`), this is the overall mean.
   - In the Bühlmann-Straub model, `c_j = w_j` and
     `w_j (1 - z_j) = w_j s² / (s² + a w_j) = (s²/a) z_j`, so `m` must be the
     credibility weighted mean `X_zw` of (8.32). This is one more reason, next
     to Remark 8.4.4 (p. 220), to estimate `m` by `X_zw` rather than by the
     ordinary weighted mean `X_ww`.

## Example 8.2.5 and an erratum

From the data of Example 8.2.1 (p. 205), the group means are 100, 110 and
120, `F = 4.588`, and `z = 1 - 1/F = 0.782`. The book gives the premiums as
102.18, 110 and 118.82. The third one should be
`0.782 · 120 + 0.218 · 110 = 117.82`; the premiums are symmetric around 110
(110 ± 7.82). `tests/test_credibility_premium.py` recomputes the example from
the original data.

## Reference check

Every result used here was located in the printed book (page numbers are the
printed ones, not the PDF page numbers) and checked as follows.

| Result | Location | Checked by hand | Checked by a test |
|---|---|---|---|
| Credibility premium `z X_j + (1 − z) X` | eq. (8.1), p. 203 | yes | `test_kaas_example_8_2_5` |
| Variance components model | eqs. (8.5)–(8.6), p. 206 | yes | `test_buhlmann_premium_is_unbiased_and_mse_optimal` |
| Optimal `z = aT/(aT + s²)` | eq. (8.10), p. 208; Theorem 8.2.4, p. 209 | yes (a = 100, s² = 64, T = 5 gives 0.8865) | same simulation test |
| Limits of `z` | Remark 8.2.3, p. 209 | yes | `test_weights_over_periods_feed_the_premium` (joint) |
| Data of the example | Example 8.2.1, p. 205 | yes | `test_kaas_example_8_2_5` (means, F = 4.588) |
| Premiums 102.18, 110, 117.82 | Example 8.2.5, p. 210 | yes: the book prints 118.82; 0.782 · 120 + 0.218 · 110 = 117.82 | `test_kaas_example_8_2_5` and the joint test with #12 |
| Virtual experience | Remark 8.2.7, eq. (8.23), p. 211 | yes (115.07 both ways) | `test_virtual_experience_remark_8_2_7` |
| `X_zw` and Bühlmann-Straub `z_j` | eq. (8.32), Theorem 8.4.1, p. 215; Remark 8.4.4, p. 220 | yes | `test_buhlmann_straub_balance_with_credibility_weighted_mean` |
| Poisson-gamma as credibility | eq. (8.60), p. 225 | yes | `test_poisson_gamma_rate_is_a_credibility_premium` |

**About the erratum.** The value 118.82 printed in Example 8.2.5 cannot be
right: with the factor 0.782 given in the same example, the premiums are
`110 ± 0.782 · 10`, that is 102.18 and 117.82, symmetric around the overall
mean 110. The example is recomputed from the original data in two
independent ways: with the ANOVA ratio `1 − 1/F`, and with Garcia's
estimators and weight from #12 (`tests/test_joint_credibility_weights.py`).
Both give 117.82.

## Independent checks (`tests/test_credibility_premium.py`)

- Example 8.2.5 recomputed from the data of Example 8.2.1.
- A simulation of the balanced Bühlmann model with 400 000 risks: the
  premium is unbiased, its MSE matches the formula above within 1 %, and
  among several values of `z` the Bühlmann factor gives the smallest MSE.
- The virtual experience identity (Remark 8.2.7) and eq. (8.60).
- Portfolio balance with `X_zw` in a random Bühlmann-Straub portfolio, and
  imbalance with `X_ww`.
- Bounds, monotonicity in `z`, components, broadcasting along an axis,
  invalid inputs and overflow.

## Joint tests with #12 (`tests/test_joint_credibility_weights.py`)

With Garcia's `estimate_process_variance`, `estimate_hypothetical_mean_variance`
and `credibility_weight` on the data of Example 8.2.1, `EPV = 108.97`,
`VHM = 100 − 108.97/5` and `z = 0.782 = 1 − 1/F`; the premium reproduces
Example 8.2.5. A second test feeds his weight for several numbers of periods
into the premium. The tests are skipped when `credibility.weights` is not in
the branch and run once #12 is merged.

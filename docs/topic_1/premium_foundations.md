# Topic 1: pure premium and the expected value principle

Functions: `expected_aggregate_loss`, `pure_premium` and
`expected_value_premium` in `therisktheory.solvency_pricing`.
Status: experimental.

Page numbers refer to Kaas, Goovaerts, Dhaene and Denuit (2008), *Modern
Actuarial Risk Theory: Using R*, 2nd edition, Springer.

## Model and conditions

Let `N` be the number of claims in a period and `X_1, X_2, ...` the claim
amounts. The aggregate claims are `S = X_1 + ... + X_N`, with `S = 0` when
there are no claims. The functions rely on these conditions:

| Condition | Where it comes from | How the code handles it |
|---|---|---|
| `X_1, X_2, ...` i.i.d. like some `X` | Section 3.1, p. 41 | Assumed; not checkable from the means |
| `N` independent of the claim amounts | Section 3.1, p. 41 | Assumed; not checkable from the means |
| `0 ≤ E[N] < ∞`, `0 ≤ E[X] < ∞` | needed for `E[S]` to exist | `ValueError` if negative, NaN or infinite |
| `0 ≤ E[S] < ∞` | follows from the above | `ValueError` if violated on input |
| `0 ≤ θ < ∞` | Section 5.3, p. 119 (`θ > 0`) | `ValueError` if violated; `θ = 0` allowed |

Nothing else is assumed about the distributions of `N` and `X`.

## Expected aggregate claims

`E[S] = E[N] E[X]`.

Given `N = n` the sum has `n` terms. Since `N` is independent of the claim
amounts, the condition can be dropped, and by linearity
`E[S | N = n] = n E[X]`. So `E[S | N] = N E[X]`, and by the law of total
expectation `E[S] = E[N E[X]] = E[N] E[X]`. This is eq. (3.3) on p. 42,
where it is written as `E[S] = μ1 E[N]` with `μ1 = E[X]`.

Two checks are in `tests/test_premiums.py`. One lists every outcome of a
small model and gets 126 = 0.7 × 180. The other simulates a compound
Poisson(3) model with exponential claims of mean 200 and compares the
sample mean with 600.

## Pure premium

The pure (net) premium is `P = E[S]`. The book calls it the net premium or
equivalence principle (Section 5.3, p. 119). It balances the expected claims
and nothing more, and the book points out that a premium without a positive
loading leads to ruin with certainty (p. 120). That is why the loaded
premiums in the rest of Topic 1 are needed.

## Expected value principle

`π[S] = (1 + θ) E[S]`, principle (b) on p. 119. The book takes `θ > 0`; the
function also accepts `θ = 0`, which gives back the pure premium.

From Section 5.3.1 and Table 5.1 (pp. 120–122), for `θ > 0`:

- The loading is non-negative, since `π[S] − E[S] = θ E[S] ≥ 0`.
- It is additive, by linearity of the expectation.
- It is proportional, `π[cS] = c π[S]` (Example 5.3.2, p. 121).
- It can charge more than the maximum loss: for a constant risk `c > 0` the
  premium is `(1 + θ)c`.
- It is not consistent, because `π[S + c] = π[S] + (1 + θ)c`.
- It is not iterative. Iterativity (property 5, p. 121) asks that
  `π[π[S | Y]] = π[S]` for every random variable `Y`: computing the premium
  in two steps, first given `Y` and then over `Y`, must give the direct
  premium. Here `π[S | Y] = (1 + θ) E[S | Y]`, so by the tower property
  `π[π[S | Y]] = (1 + θ) E[(1 + θ) E[S | Y]] = (1 + θ)² E[S]`, which differs
  from `π[S] = (1 + θ) E[S]` whenever `θ > 0` and `E[S] > 0`. For a constant
  risk this is the same as loading an already loaded premium a second time.

The main weakness is that the loading only depends on the mean, so two risks
with the same mean and very different variances pay the same premium. The
variance and standard deviation principles (also on p. 119) deal with that.

## Numerical behaviour

- Scalars give 0-d float64 arrays; arrays are broadcast with the NumPy rules
  and empty arrays give empty results.
- Inputs must be real numbers (integers or floats). Booleans, complex
  numbers, strings, `None` and ragged lists raise `TypeError`.
- Shapes that cannot be broadcast together raise `ValueError`.
- With valid inputs the result can still overflow, for example
  `1e200 × 1e200`. The product is computed with NumPy's overflow warning
  silenced and an `OverflowError` is raised instead, so the caller never
  gets `inf` back.

## Imports

The module only needs NumPy at runtime. `numpy.typing` is imported under
`TYPE_CHECKING` because it is only used in annotations, and SciPy is not
imported at all.

# Topic 2: fair games, unfair games and the gambler's ruin

Functions: `fair_game_expectation`, `unfair_game_expectation`,
`gambler_ruin_probability` and `simulate_gambler_ruin` in
`therisktheory.finetti_ruin`. Status: experimental.

## Fair and unfair games

A game pays the player `y_i` with probability `p_i` in exchange for a stake
`c`. The expected net gain per round is `E[Y] - c`, and over `n` independent
rounds it is `n (E[Y] - c)`. The game is fair when this is zero, so the fair
stake is the mathematical expectation `E[Y] = sum_i p_i y_i`.

| Condition | How the code handles it |
|---|---|
| `0 <= p_i <= 1`, `sum_i p_i = 1` (tolerance `1e-9`) | `ValueError` |
| payoffs finite real numbers (may be negative) | `ValueError` / `TypeError` |
| stake finite, `n >= 0` integer | `ValueError` / `TypeError` |

The link with Topic 1 is direct. If the insured pays the pure premium
`E[S]` and receives the claim `S`, the contract is a fair game. With the
expected value premium `(1 + θ) E[S]` the insured's expected net gain is
`-θ E[S]`, so the game is unfair for the insured and favourable for the
insurer. This is tested in `tests/test_games.py`.

## Gambler's ruin

A player starts with `z` units and bets one unit per round. Each round is
won with probability `p` and lost with probability `q = 1 - p`,
independently. The game ends at `0` (ruin) or at a target `a`.

**Conditions.** `z` and `a` are whole numbers of stakes with `0 <= z <= a`,
`a` may be infinite, and `0 <= p <= 1`.

**Derivation.** Let `P_z` be the ruin probability from `z`. Conditioning on
the first round,

```text
P_z = p P_{z+1} + q P_{z-1},   0 < z < a,   P_0 = 1,   P_a = 0.
```

Trying `P_z = λ^z` gives `p λ² - λ + q = 0`, with roots `λ = 1` and
`λ = r = q / p`.

- If `p ≠ q` the roots differ, `P_z = A + B r^z`, and the boundary
  conditions give `P_z = (r^z - r^a) / (1 - r^a)`.
- If `p = q` the root `1` is double, `P_z = A + B z`, and `P_z = 1 - z / a`.

One can check directly that both expressions satisfy the recurrence and
the boundary conditions. Letting `a → ∞`: if `p > q`, then `r < 1` and `P_z → r^z`; if `p ≤ q` the
ruin is certain.

**Numerics.** The formulas are algebraically equal to the ones above, but
each case is written so that two nearly equal numbers are never subtracted:

| Case | Form used | Problem it avoids |
|---|---|---|
| `p > q` (`r < 1`) | `r^z (1 - r^(a-z)) / (1 - r^a)` | `r^z - r^a` cancels when both are tiny, so `1e-18` became `0.0` |
| `p < q` (`r > 1`) | `(1 - s^(a-z)) / (1 - s^a)`, `s = 1/r` | overflow of `r^a` for large `a` |
| `p = q` | `(a - z) / a` | `1 - z/a` loses digits when `z` is close to `a` |

The differences `1 - r^k` are computed with `expm1`, and `log r` with
`log1p(-p) - log(p)`, so the result also stays accurate when `p` is close to
`1/2`.

The `p > q` case was found in the peer review of PR #3: for `p = 0.6`,
`z = 100`, `a = 1000` the old form returned `0.0` instead of
`2.4596544265798527e-18`. `examples/gambler_ruin.py` reproduces it, and
`tests/test_games.py` compares the function with exact rational arithmetic
(`fractions.Fraction`) using `atol=0`, so that a result of `0.0` cannot pass
as an approximation of a small probability.

## Relation with risk theory

With an infinite target the gambler's fortune is a discrete-time risk
process `U(n) = u + G_1 + ... + G_n` with `G_i = ±1`, the model of Kaas et
al. (2008), Section 4.5, p. 98.

- The adjustment coefficient solves `m_G(-R) = p e^{-R} + q e^{R} = 1`,
  which gives `R = log(p / q)`, positive exactly when the game is
  favourable to the player (`p > q`, a positive safety loading).
- Kaas et al. define ruin as `U < 0`. With unit steps that is the gambler's
  ruin from `z = u + 1`, so `ψ(u) = r^{u+1} = e^{-Ru} / e^{R}`. This is exactly
  eq. (4.36) on p. 98, because the capital at ruin is always `U(T) = -1`.
  The bound `ψ(u) ≤ e^{-Ru}` is eq. (4.37).
- In a fair game `p = q` there is no positive `R` and ruin is certain, the
  same conclusion as for a premium without loading (Topic 1).
- Remark 4.4.3 (p. 97) interprets `e^{-R U(t)}` as the fortune of a gambler
  in a sequence of fair games.

These relations are checked numerically in `tests/test_games.py`.

## Independent checks

- The exact formula is compared with a direct solution of the linear system
  of the recurrence (`numpy.linalg.solve`).
- Symmetry: the gambler's ruin is the opponent's win,
  `P_z(p) = 1 - P_{a-z}(1 - p)`.
- In a fair game the expected final fortune `a (1 - P_z)` equals `z`.
- `simulate_gambler_ruin` is compared with the formula (5 standard errors),
  and its mean duration with the expected duration `D_z`. Conditioning on
  the first round gives `D_z = 1 + p D_{z+1} + q D_{z-1}`, `D_0 = D_a = 0`,
  which the tests solve as a linear system. In a fair game `D_z = z (a - z)`,
  which can be checked by substitution.

## References

- Kaas, R., Goovaerts, M., Dhaene, J. and Denuit, M. (2008). *Modern
  Actuarial Risk Theory: Using R*, 2nd ed. Springer. Remark 4.4.3, p. 97;
  Section 4.5, eqs. (4.33)–(4.37), p. 98.

The ruin probability itself is not taken from a book: it follows from the
derivation above, which the tests also check against a direct numerical
solution of the recurrence.

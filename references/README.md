# References

Record bibliographic sources used to validate mathematical implementations here.

Do not upload copyrighted books or papers unless redistribution is permitted.

Prefer bibliographic metadata and stable links/DOIs where available.

## Sources

### Kaas, Goovaerts, Dhaene and Denuit (2008)

Kaas, R., Goovaerts, M., Dhaene, J. and Denuit, M. (2008). *Modern Actuarial Risk Theory:
Using R*, 2nd ed. Springer. ISBN 978-3-540-70992-3, e-ISBN 978-3-540-70998-5.

| Used for | Location |
|---|---|
| Collective model and independence assumptions | Section 3.1, eq. (3.1), p. 41 |
| `expected_aggregate_loss`: `E[S] = E[N] E[X]` | Section 3.2, eqs. (3.2)–(3.3), pp. 42–43 |
| `pure_premium`: net premium / equivalence principle | Section 5.3, principle (a), p. 119 |
| `expected_value_premium`: `(1 + θ) E[S]` | Section 5.3, principle (b), p. 119 |
| Properties of the expected value principle | Section 5.3.1, pp. 120–121; Table 5.1, p. 122 |
| Martingale as the fortune of a gambler in fair games (Topic 2) | Remark 4.4.3, p. 97 |
| Discrete-time ruin model, adjustment coefficient, eqs. (4.36)–(4.37) (Topic 2) | Section 4.5, p. 98 |
| Poisson and negative binomial claim numbers, overdispersion (Topic 3) | Section 3.3, p. 45; Section 3.9.4, eq. (3.74), p. 64; Table A, p. 368 |
| Mixed Poisson, structure variable, gamma mixture gives negative binomial (Topic 3) | Example 3.3.1, eqs. (3.13)–(3.15), pp. 45–46 |
| mgf of a compound sum `m_S(t) = m_N(log m_X(t))` (Topic 3) | Section 3.2, eq. (3.5), p. 43 |
| Sum of compound Poisson variables is compound Poisson (Topic 3) | Theorem 3.4.1, eq. (3.20), p. 47 |
| Poisson process, intensity, infinitesimal increments (Topic 3) | Definition 4.2.1, eqs. (4.4)–(4.5), p. 89 |
| Gamma posterior and predicted claim rate `(α + x_Σ)/(τ + T)` (Topic 3) | Section 8.5, eq. (8.59), p. 225 |

Modules: `therisktheory.solvency_pricing.premiums` (Topic 1),
`therisktheory.finetti_ruin.games` (Topic 2) and `therisktheory.frequency`
(Topic 3).

The gambler's ruin probability (Topic 2) is justified by the derivation in
`docs/topic_2/gambler_ruin.md`, not by a book reference.

# References

Record bibliographic sources used to validate mathematical implementations here.

Do not upload copyrighted books or papers unless redistribution is permitted.

Prefer bibliographic metadata and stable links/DOIs where available.

## Sources

### Kaas, Goovaerts, Dhaene and Denuit (2008)

Kaas, R., Goovaerts, M., Dhaene, J. and Denuit, M. (2008). *Modern Actuarial Risk Theory:
Using R*, 2nd ed. Springer. ISBN 978-3-540-70992-3, e-ISBN 978-3-540-70998-5.
DOI: [10.1007/978-3-540-70998-5](https://doi.org/10.1007/978-3-540-70998-5)
(metadata checked against Crossref).

| Used for | Location |
|---|---|
| Collective model and independence assumptions | Section 3.1, eq. (3.1), p. 41 |
| `expected_aggregate_loss`: `E[S] = E[N] E[X]` | Section 3.2, eqs. (3.2)–(3.3), pp. 42–43 |
| `pure_premium`: net premium / equivalence principle | Section 5.3, principle (a), p. 119 |
| `expected_value_premium`: `(1 + θ) E[S]` | Section 5.3, principle (b), p. 119 |
| Properties of the expected value principle | Section 5.3.1, pp. 120–121; Table 5.1, p. 122 |
| Martingale as the fortune of a gambler in fair games (Topic 2) | Remark 4.4.3, p. 97 |
| Discrete-time ruin model, adjustment coefficient, eqs. (4.36)–(4.37) (Topic 2) | Section 4.5, p. 98 |
| Credibility premium `z X_j + (1 - z) X`, objectives of credibility theory (Topic 4) | Section 8.1, eq. (8.1), pp. 203–204 |
| Balanced Bühlmann model, variance components (Topic 4) | Section 8.2, eqs. (8.5)–(8.6), p. 206 |
| Optimal credibility premium and factor `z = aT/(aT + s²)` (Topic 4) | Theorems 8.2.2 and 8.2.4, eqs. (8.7)–(8.16), pp. 207–209 |
| Asymptotic properties of `z` (Topic 4) | Remark 8.2.3, p. 209 |
| Worked example and shrinkage (Topic 4) | Examples 8.2.1 and 8.2.5, pp. 205 and 210 |
| Virtual experience (Topic 4) | Remark 8.2.7, eq. (8.23), p. 211 |
| Bühlmann-Straub model, credibility weighted mean `X_zw` (Topic 4) | Eq. (8.32), Theorem 8.4.1, p. 215; Remark 8.4.4, p. 220 |

Modules: `therisktheory.solvency_pricing.premiums` (Topic 1),
`therisktheory.finetti_ruin.games` (Topic 2) and
`therisktheory.credibility.premium` (Topic 4).

The gambler's ruin probability (Topic 2) is justified by the derivation in
`docs/topic_2/gambler_ruin.md`, not by a book reference.

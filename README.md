# TheRiskTheory

**A human-validated, vectorized Python implementation of Mathematical Risk Theory.**

TheRiskTheory is an educational and scientific open-source project created around a Mathematical Risk Theory II course. Its goal is to transform the complete mathematical content of the subject into reusable, tested, documented Python functions.

> **AI-assisted development. Human-validated mathematics.**

AI tools may be used to help write code, tests, examples, and documentation. AI output is never considered a mathematical source and cannot independently validate a formula or implementation.

## Design principles

1. **Python first**
2. **NumPy is the canonical computational backend**
3. **SciPy is used for probability distributions and numerical methods**
4. **Vectorized implementations are preferred**
5. **Mathematics is independent from DataFrame libraries**
6. **pandas, Polars, and Spark compatibility is implemented through optional adapters**
7. **Every mathematical implementation requires tests, documentation, references, and human review**

## Installation

```bash
git clone https://github.com/<owner>/therisktheory.git
cd therisktheory
pip install -e ".[dev]"
```

Optional backends:

```bash
pip install -e ".[pandas]"
pip install -e ".[polars]"
pip install -e ".[spark]"
```

## Quick example

```python
import numpy as np
from therisktheory.solvency_pricing import pure_premium

frequency = np.array([0.5, 1.0, 2.0])
mean_severity = np.array([1000.0, 750.0, 500.0])

premium = pure_premium(frequency, mean_severity)
print(premium)
```

## Course architecture

The syllabus is divided into six development areas so that six contributors can work in parallel.

| Area | Scope |
|---|---|
| `solvency_pricing` | Insurer solvency, pure premium, safety loading, guarantee capital, retention, reinsurance |
| `finetti_ruin` | Fair and unfair games, De Finetti theory, safety index, reserve fund, ruin probability |
| `frequency` | Claim counts, claim intensity, Poisson, Negative Binomial, compound Poisson, Pólya processes |
| `severity_aggregate` | Lognormal, Pareto, Weibull, Beta, aggregate losses, approximations, Esscher, Monte Carlo |
| `credibility` | Full and partial credibility, Bayesian credibility, credibility weights, claim reserves |
| `collective_risk` | Collective risk process, insolvency, Lundberg equation, adjustment coefficient, reinsurance effects |

See [`docs/SYLLABUS.md`](docs/SYLLABUS.md) for the full mapping.

## Repository structure

```text
therisktheory/
├── src/
│   └── therisktheory/
│       ├── solvency_pricing/
│       ├── finetti_ruin/
│       ├── frequency/
│       ├── severity_aggregate/
│       ├── credibility/
│       ├── collective_risk/
│       └── backends/
├── tests/
├── docs/
├── examples/
├── notebooks/
├── references/
├── .github/
├── CONTRIBUTING.md
├── pyproject.toml
└── LICENSE
```

## Canonical backend

The mathematical reference implementation must use NumPy/SciPy whenever reasonably possible.

```python
def mathematical_function(x):
    x = np.asarray(x, dtype=float)
    ...
    return result
```

The mathematical core must **not** require pandas, Polars, or Spark.

Future backend adapters must reproduce the NumPy reference behavior:

```text
NumPy reference
      |
      +-- pandas adapter
      +-- Polars adapter
      +-- Spark expression / Arrow execution
```

Backend parity should satisfy, within documented numerical tolerances:

```text
numpy_result ≈ pandas_result ≈ polars_result ≈ spark_result
```

## Contribution unit

Each student contribution should implement at least one complete mathematical function.

A contribution is not complete until it contains:

- mathematical definition;
- assumptions and domain restrictions;
- vectorized Python implementation;
- type hints;
- Google-style docstring;
- bibliographic reference;
- analytical unit test;
- boundary/invalid-input tests;
- numerical validation where appropriate;
- example;
- peer mathematical review.

## Validation states

Every implemented concept should have one of these states:

```text
experimental
reviewed
mathematically_validated
```

### `experimental`
Implementation exists but has not completed mathematical review.

### `reviewed`
Code and derivation have been reviewed by another contributor.

### `mathematically_validated`
The formula, implementation, tests, assumptions, and references have been approved by the instructor/maintainer.

No contributor should mathematically validate their own implementation.

## Performance rules

Prefer:

```python
np.exp(x)
np.sum(x, axis=...)
np.where(...)
scipy.stats...
```

over Python row-by-row loops.

Avoid:

```python
for row in dataframe:
    ...
```

inside mathematical implementations.

Spark implementations must not collect large datasets to the driver. Use Spark-native expressions when possible and Arrow/Pandas UDF execution only when necessary.

## Testing

```bash
pytest
```

With coverage:

```bash
pytest --cov=therisktheory --cov-report=term-missing
```

## Formatting and linting

```bash
ruff check .
ruff format .
```

The style rules are configured in `pyproject.toml` and checked in CI:

- PEP 8 and the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html);
- lines of at most 80 characters;
- Google-style docstrings (ruff `D` rules with `convention = "google"`);
- absolute imports, sorted by ruff;
- CI runs `ruff check .`, `ruff format --check .` and the tests with line and
  branch coverage.

## Governance

TheRiskTheory uses an open contribution model: anyone may contribute through pull requests.

The `master` branch is protected and **all merges require final approval from `@luispitta`**.

See [`GOVERNANCE.md`](GOVERNANCE.md) for the complete repository policy.

## License

TheRiskTheory is licensed under the **Apache License 2.0**. See [`LICENSE`](LICENSE).

Unless explicitly stated otherwise, contributions submitted to this repository are provided under the same license.

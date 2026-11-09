"""Joint tests: credibility premium (#18) with Garcia's weights (#12).

These tests use ``credibility.weights`` from #12. They are skipped when that
module is not in the branch, so this PR does not fail because of another PR;
once both PRs are merged they run against the real code.
"""

import numpy as np
import pytest

from therisktheory import credibility

weights = pytest.importorskip(
    "therisktheory.credibility.weights",
    reason="needs Garcia's credibility weights (#12)",
)

# Kaas et al. (2008), Example 8.2.1, p. 205.
KAAS_821 = np.array(
    [
        [99.3, 93.7, 103.9, 92.5, 110.6],
        [112.5, 108.3, 118.0, 99.4, 111.8],
        [129.2, 140.9, 108.3, 105.0, 116.6],
    ]
)


def test_kaas_example_with_estimated_weights() -> None:
    # Garcia's estimators: EPV = MSW and VHM = MSB/T - MSW/T, so his weight
    # T VHM / (T VHM + EPV) equals 1 - MSW/MSB = 1 - 1/F = 0.782 of
    # Example 8.2.5 (p. 210). The premium then reproduces the example.
    n_years = KAAS_821.shape[1]
    epv = weights.estimate_process_variance(KAAS_821)
    vhm = weights.estimate_hypothetical_mean_variance(KAAS_821)
    z = weights.credibility_weight(n_years, epv, vhm)
    assert float(epv) == pytest.approx(108.97, abs=5e-3)
    assert float(vhm) == pytest.approx(100.0 - 108.97 / 5, abs=5e-3)
    assert float(z) == pytest.approx(0.782, abs=5e-4)

    group_means = KAAS_821.mean(axis=1)
    premium = credibility.credibility_premium(group_means, KAAS_821.mean(), z)
    np.testing.assert_allclose(premium, [102.18, 110.0, 117.82], atol=0.01)

    props = credibility.credibility_premium_properties(
        group_means, KAAS_821.mean(), z
    )
    assert props["within_bounds"].all()
    assert float(props["portfolio_imbalance"]) == pytest.approx(0.0, abs=1e-10)


def test_weights_over_periods_feed_the_premium() -> None:
    # The weight grows with the number of periods (Remark 8.2.3) and the
    # premium moves from the collective mean towards the individual one.
    n = np.array([0, 1, 5, 20, 1000])
    z = weights.credibility_weight(n, 64.0, 100.0)
    premium = credibility.credibility_premium(117.0, 100.0, z)
    assert premium[0] == 100.0
    assert np.all(np.diff(premium) > 0)
    assert premium[-1] == pytest.approx(117.0, abs=0.02)
    np.testing.assert_allclose(z, 100.0 * n / (100.0 * n + 64.0))

import numpy as np

from therisktheory import credibility

# Kaas et al. (2008), Example 8.2.1: three groups observed for five years.
claims = np.array(
    [
        [99.3, 93.7, 103.9, 92.5, 110.6],
        [112.5, 108.3, 118.0, 99.4, 111.8],
        [129.2, 140.9, 108.3, 105.0, 116.6],
    ]
)
group_means = claims.mean(axis=1)  # [100. 110. 120.]
overall_mean = claims.mean()  # 110.0
z = 0.782  # Example 8.2.5; normally from credibility_weight (#12)

print(credibility.credibility_premium(group_means, overall_mean, z))
# [102.18 110.   117.82]

parts = credibility.credibility_components(group_means, overall_mean, z)
print(parts["experience_correction"])  # [-7.82  0.    7.82]

props = credibility.credibility_premium_properties(group_means, overall_mean, z)
print(props["within_bounds"], props["portfolio_imbalance"])
# [ True  True  True] 0.0

import numpy as np

from therisktheory import finetti_ruin

# A player with 5 units plays until ruin or until reaching 10 units.
p = np.array([0.45, 0.50, 0.55])
print(finetti_ruin.gambler_ruin_probability(5, 10, p))
# [0.7317174 0.5       0.2682826]

# Against an unlimited opponent only a favourable game avoids certain ruin.
print(finetti_ruin.gambler_ruin_probability(5, np.inf, p))
# [1.         1.         0.36664783]

sim = finetti_ruin.simulate_gambler_ruin(5, 10, 0.45, n_paths=20_000, rng=1)
print(f"{sim.ruin_probability:.3f} +- {sim.standard_error:.3f}")

# Buying insurance at a 15 % loading is an unfair game for the insured.
probs, claims = [0.9, 0.08, 0.02], [0.0, 1000.0, 20000.0]
fair = finetti_ruin.fair_game_expectation(probs, claims)
print(fair, finetti_ruin.unfair_game_expectation(probs, claims, 1.15 * fair))
# 480.0 -72.0

# Very small ruin probabilities are kept, not rounded to 0 (review of #3).
print(finetti_ruin.gambler_ruin_probability(100, 1000, 0.6))
# 2.4596544265798527e-18
print(finetti_ruin.gambler_ruin_probability(30, 1000, 0.9))
# 2.3589824875925706e-29

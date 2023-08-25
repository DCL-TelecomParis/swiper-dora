from fractions import Fraction
from typing import List, Union

from solver.general_solver import Solution
from solver.knapsack import knapsack
from solver.antiknapsack import antiknapsack


class WeightRamp:

    """Represents an instance of the Weight Ramp problem."""

    def __init__(self,
                 weights: List[Union[Fraction, float, int]],
                 alpha_w: Union[Fraction, float],
                 beta_w: Union[Fraction, float]):
        """
        Create a new instance.

        :param weights: list of weights of the parties
        :param alpha_w: low weighted threshold
        :param beta_w: high weighted threshold
        """
        # Number of parties
        self.n = len(weights)
        # List of weights of the parties
        self.weights = weights
        # Total weight of all parties
        self.total_weight = sum(weights)
        self.low_threshold_weight = alpha_w * self.total_weight
        self.high_threshold_weight = beta_w * self.total_weight
        self.alpha_w = alpha_w
        self.beta_w = beta_w

    def __str__(self):
        return f"WeightRamp < " \
               f"n={self.n}, weights=[{' '.join(map(str, self.weights))}], alpha_w={self.alpha_w}, beta_w={self.beta_w}" \
               f" >"

    def __repr__(self):
        return str(self)

    def check_solution(self, sol: Solution, no_jit: bool) -> bool:
        _, alpha_t = knapsack(self.weights, sol.values, self.low_threshold_weight,
                              upper_bound=self.beta_w * sol.sum + 1, return_set=False, no_jit=no_jit)
        _, beta_t = antiknapsack(self.weights, sol.values, self.high_threshold_weight,
                                 lower_bound=self.alpha_w * sol.sum, return_set=False, no_jit=no_jit)
        return alpha_t < beta_t

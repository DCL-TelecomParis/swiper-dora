"""
While knapsack finds the best (in terms of the number of tickets) possible set of weight at most capacity,
antiknapsack finds the worst possible set of weight at least capacity.
It can be trivially reduced to knapsack by searching for the complement of the solution.
"""

from fractions import Fraction
from typing import List, Union

from solver.knapsack import knapsack_upper_bound, knapsack


def antiknapsack(
        weights: List[Union[Fraction, float, int]],
        profits: List[int],
        capacity: Union[Fraction, float, int],
        lower_bound: int,
        no_jit: bool) -> Tuple[List[int], int]:
    profits_sum = sum(profits)
    best_complement, best_complement_profit = knapsack(weights, profits, sum(weights) - capacity, profits_sum - lower_bound, no_jit)
    best_complement = set(best_complement)
    return [i for i in range(len(weights)) if i not in best_complement], profits_sum - best_complement_profit


def antiknapsack_lower_bound(
        weights: List[Union[Fraction, float, int]],
        profits: List[int],
        capacity: Union[Fraction, float, int],
) -> int:
    return sum(profits) - knapsack_upper_bound(weights, profits, sum(weights) - capacity)

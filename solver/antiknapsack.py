"""
While knapsack finds the best (in terms of the number of tickets) possible set of weight at most capacity,
antiknapsack finds the worst possible set of weight at least capacity.
It can be trivially reduced to knapsack by searching for the complement of the solution.
"""

from fractions import Fraction
from typing import List, Union, Tuple

from solver.knapsack import knapsack_upper_bound, knapsack


def antiknapsack(
        weights: List[Union[Fraction, float, int]],
        profits: List[int],
        capacity: Union[Fraction, float, int],
        lower_bound: int,
        return_set: bool,
        no_jit: bool) -> Tuple[List[int], int]:
    profits_sum = sum(profits)
    best_complement, best_complement_profit = knapsack(weights, profits, capacity=sum(weights) - capacity,
                                                       upper_bound=profits_sum - lower_bound,
                                                       return_set=return_set, no_jit=no_jit)

    worst_set = None
    if return_set:
        best_complement = set(best_complement)
        worst_set = [i for i in range(len(weights)) if i not in best_complement]

    worst_profit = profits_sum - best_complement_profit
    return worst_set, worst_profit


def antiknapsack_lower_bound(
        weights: List[Union[Fraction, float, int]],
        profits: List[int],
        capacity: Union[Fraction, float, int],
) -> int:
    return sum(profits) - knapsack_upper_bound(weights, profits, sum(weights) - capacity)

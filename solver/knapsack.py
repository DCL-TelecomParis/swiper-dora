import inspect
import logging
from fractions import Fraction
from typing import Union, Tuple, List, Optional
from numba import njit
import numpy as np
from numba.core.extending import register_jitable

from solver.util import lcm, rev

overflow_warning = ("Integer overflow while converting fractional weights to integers for JIT compilation. "
                    "Using pure python implementation, which is much slower for large instances.")

MAX_INT_64 = np.iinfo(np.int64).max


def knapsack(
        weights: List[Union[Fraction, float, int]],
        profits: List[int],
        capacity: Union[Fraction, float, int],
        upper_bound: int,
        return_set: bool,
        no_jit: bool) -> Union[Tuple[List[int], int], int]:
    """
    Solves the given knapsack instance up to the given upper bound on profit.
    Finds the set of items with the highest profit that fits into the knapsack of the given capacity if its profit
    is upper_bound or less.
    Otherwise, finds any set of items that fits into the knapsack and has profit greater than upper_bound.

    If return_set is True, returns the set of items and its profit or upper_bound + 1 if it exceeds the upper bound.
    Otherwise, returns only the profit.

    Running time: O(len(weights) * upper_bound).
    Memory usage: O(len(weights)) if return_set is False, O(len(weights) * upper_bound) otherwise.

    NB: the memory footprint can be reduced from O(len(weights) * upper_bound) to O(len(weights) + upper_bound).
    See: Section 3.3 of "Knapsack problems" by Pisinger, D., & Toth, P. (1998)
    """
    assert len(weights) > 0

    if no_jit:
        res = _knapsack_impl(weights, profits, capacity, upper_bound, return_set)
        return res if return_set else res[1]

    if isinstance(weights[0], float):
        assert isinstance(capacity, float)

        # Call the JIT-compiled function
        res = _knapsack_jit_float(np.array(weights, dtype=np.float64), np.array(profits, dtype=np.int64),
                                  capacity, upper_bound, return_set)
        return res if return_set else res[1]

    if isinstance(weights[0], int):
        # Capacity may be a fraction. However, rounding it down does not affect the result.
        capacity = int(capacity)

        # Make sure that all integers fit into 64 bits to avoid overflows
        if sum(weights) > MAX_INT_64 or sum(profits) > MAX_INT_64 or capacity > MAX_INT_64:
            logging.warning(overflow_warning)
            res = _knapsack_impl(weights, profits, capacity, upper_bound, return_set)
            return res if return_set else res[1]

        # Call the JIT-compiled function
        res = _knapsack_jit_int(np.array(weights, dtype=np.int64), np.array(profits, dtype=np.int64),
                                capacity, upper_bound, return_set)
        return res if return_set else res[1]

    if isinstance(weights[0], Fraction):
        assert isinstance(capacity, Fraction)

        # Normalize the weights to integers
        denominator_lcm = lcm(w.denominator for w in weights)
        weights = [int(w * denominator_lcm) for w in weights]
        # solve the problem with integer weights
        return knapsack(weights, profits, capacity * denominator_lcm, upper_bound, return_set, no_jit)

    raise ValueError(f"Unsupported type {type(weights[0])} for weights")


@register_jitable
def _knapsack_impl(weights, profits, capacity, upper_bound, return_set) -> Tuple[Optional[List[int]], int]:

    assert len(weights) == len(profits)

    # If any item has profit greater than the upper bound, just return it.
    # In the rest of the code, we assume that all profits are at most upper_bound.
    for i in range(len(weights)):
        if profits[i] > upper_bound:
            return [i] if return_set else None, profits[i]

    # Ignore items with zero profit
    nonzero_items = [i for i in range(len(weights)) if profits[i] > 0]
    n_nonzero = len(nonzero_items)

    # after i-th iteration of the loop, dp[i] is the minimum weight of a subset of items 0, ..., i
    # with profit at least q.
    dp: List[int] = [0 if q == 0 else MAX_INT_64 for q in range(upper_bound + 2)]

    # take_item[i][q] is True iff the optimal solution for the sub-problem with items 0, ..., i
    # and profit at least q contains item i.
    # We only need it if we want to recover the set of items in the optimal solution.
    take_item: List[List[bool]] = None if not return_set else [[False] * (upper_bound + 2)] * n_nonzero

    # Fill in the table
    for i in range(n_nonzero):
        item = nonzero_items[i]

        # Update the `dp` from right to left to avoid overwriting the values that we still need
        for q in rev(range(upper_bound + 2)):
            weight_if_item_taken = MAX_INT_64
            if profits[item] >= q:
                weight_if_item_taken = weights[item]
            elif dp[q - profits[item]] != MAX_INT_64:
                weight_if_item_taken = dp[q - profits[item]] + weights[item]

            if weight_if_item_taken < dp[q]:
                dp[q] = weight_if_item_taken
                if return_set:
                    take_item[i][q] = True

    # Solution is the maximum index of y that does not surpass capacity
    opt_value = max([q for q in range(upper_bound + 2) if dp[q] <= capacity])

    opt_set = None
    if return_set:
        # Backtrack to find the items
        opt_set = []
        q = opt_value
        for i in rev(range(n_nonzero)):
            if take_item[i][q]:
                opt_set.append(nonzero_items[i])
                q -= profits[nonzero_items[i]]
            if q <= 0:
                break
        assert q <= 0
        # [::-1] reverses the list so that the items are sorted in increasing order
        opt_set = opt_set[::-1]

    return opt_set, opt_value


@njit
def _knapsack_jit_float(
        weights: np.array,
        profits: np.array,
        capacity: float,
        upper_bound: int,
        return_set: bool) -> Tuple[Optional[List[int]], int]:
    return _knapsack_impl(weights, profits, capacity, upper_bound, return_set)


@njit
def _knapsack_jit_int(
        weights: np.array,
        profits: np.array,
        capacity: np.int64,
        upper_bound: int,
        return_set: bool) -> Tuple[Optional[List[int]], int]:
    return _knapsack_impl(weights, profits, capacity, upper_bound, return_set)


def sanity_knapsack(
        weights: List[Union[Fraction, float, int]],
        profits: List[int],
        capacity: Union[Fraction, float, int],
        upper_bound: int
) -> int:
    """ Solve knapsack using dynamic programming by profits."""
    n = len(weights)
    assert len(profits) == n

    y = [0] + [sum(weights) for _ in range(1, upper_bound + 1)]

    # Fill the table
    for j in range(n):
        for q in range(upper_bound, profits[j] - 1, -1):
            if y[q - profits[j]] + weights[j] < y[q]:
                y[q] = y[q - profits[j]] + weights[j]

    # Solution is the maximum index of y that does not surpass capacity
    return max([q for q in range(upper_bound + 1) if y[q] <= capacity])


def knapsack_upper_bound(
        weights: List[Union[Fraction, float, int]],
        profits: List[int],
        capacity: Union[Fraction, float, int],
) -> int:
    """
    Returns an upper bound for the knapsack solution in quasilinear time.

    NB: this upper bound can be computed in linear time using a slightly more complicated algorithm.
    See: Section 3.1 of "Knapsack problems" by Pisinger, D., & Toth, P. (1998)
    """

    n = len(weights)
    assert len(profits) == n

    descending_efficiency_parties = sorted(range(n), key=lambda i: profits[i] / weights[i], reverse=True)

    profit = 0
    for party in descending_efficiency_parties:
        if capacity >= weights[party]:
            capacity -= weights[party]
            profit += profits[party]
        else:
            profit += profits[party] * (capacity / weights[party])
            break

    return profit

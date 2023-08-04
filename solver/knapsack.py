import inspect
import math
from fractions import Fraction
from typing import Union, Tuple, List
from numba import njit
import numpy as np

overflow_error = ValueError("Integer overflow while converting fractional weights to integers for JIT compilation. "
                            "Use --no-jit to use pure python implementation or --float to use floating point "
                            "arithmetic.")

MAX_INT_64 = np.iinfo(np.int64).max


def knapsack(
        weights: List[Union[Fraction, float, int]],
        profits: List[int],
        capacity: Union[Fraction, float, int],
        upper_bound: int,
        no_jit: bool) -> Tuple[List[int], int]:
    """
    Return the optimal set of items for the given knapsack problem.

    Takes time O(len(weights) * upper_bound).

    NB: the memory footprint can be reduced from O(len(weights) * upper_bound) to O(len(weights) + upper_bound).
    See: Section 3.3 of "Knapsack problems" by Pisinger, D., & Toth, P. (1998)
    """
    assert len(weights) > 0

    if no_jit:
        return _knapsack_impl(weights, profits, capacity, upper_bound)

    if isinstance(weights[0], float):
        assert isinstance(capacity, float)

        # Call the JIT-compiled function
        return _knapsack_jit_float(np.array(weights, dtype=np.float64), np.array(profits, dtype=np.int64),
                                   capacity, upper_bound)

    if isinstance(weights[0], int):
        # If capacity is a fraction, normalize the weights and the capacity to integers
        if isinstance(capacity, Fraction):
            if capacity.numerator != 0 and capacity.denominator != 1:
                gcd = math.gcd(capacity.numerator, capacity.denominator)
                if capacity.denominator != gcd:
                    weights = [w * (capacity.denominator // gcd) for w in weights]
                    capacity = int(capacity * (capacity.denominator // gcd))
        capacity = int(capacity)

        # Make sure that all integers fit into 64 bits to avoid overflows
        if sum(weights) > MAX_INT_64 or sum(profits) > MAX_INT_64 or capacity > MAX_INT_64:
            raise overflow_error

        # Call the JIT-compiled function
        return _knapsack_jit_int(np.array(weights, dtype=np.int64), np.array(profits, dtype=np.int64),
                                 capacity, upper_bound)

    if isinstance(weights[0], Fraction):
        assert isinstance(capacity, Fraction)

        # Normalize the weights and the capacity to integers
        lcm = capacity.denominator
        for w in weights:
            if lcm % w.denominator != 0:
                lcm = lcm * (w.denominator // math.gcd(lcm, w.denominator))

        weights = [int(w * lcm) for w in weights]
        capacity = int(capacity * lcm)

        # Check that the weights and capacity fit into 64-bit integers
        if any(w > MAX_INT_64 for w in weights) or capacity > MAX_INT_64:
            raise overflow_error

        # Call the JIT-compiled function
        return _knapsack_jit_int(np.array(weights, dtype=np.int64), np.array(profits, dtype=np.int64),
                                 capacity, upper_bound)

    raise ValueError(f"Unsupported type {type(weights[0])} for weights")


def _knapsack_impl(
        weights: List[Union[Fraction, float, int]],
        profits: List[int],
        capacity: Union[Fraction, float, int],
        upper_bound: int) -> Tuple[List[int], int]:
    total_weight = sum(weights)
    table_parties = [i for i in range(len(weights)) if profits[i] > 0]
    n_holders = len(table_parties)

    fit = [[False] * (upper_bound + 2) for _ in range(n_holders + 1)]
    dp: List[List[int]] = [([0] + [total_weight] * (upper_bound + 1)) for _ in range(n_holders + 1)]

    # Fill the table
    for j in range(1, n_holders + 1):
        for q in range(upper_bound + 1, profits[table_parties[j - 1]] - 1, -1):
            if dp[j - 1][q - profits[table_parties[j - 1]]] + weights[table_parties[j - 1]] < dp[j - 1][q]:
                dp[j][q] = dp[j - 1][q - profits[table_parties[j - 1]]] + weights[table_parties[j - 1]]
                fit[j][q] = True
            else:
                dp[j][q] = dp[j - 1][q]
        for q in range(min(profits[table_parties[j - 1]], upper_bound + 1)):
            if weights[table_parties[j - 1]] < dp[j - 1][q]:
                dp[j][q] = weights[table_parties[j - 1]]
                fit[j][q] = True
            else:
                dp[j][q] = dp[j - 1][q]

    # Solution is the maximum index of y that does not surpass capacity
    opt_value = max([q for q in range(upper_bound + 2) if dp[n_holders][q] <= capacity])

    # Backtrack to find the items
    opt_set = []
    q = opt_value
    for j in range(n_holders - 1, 0, -1):
        if fit[j][q]:
            opt_set.append(table_parties[j - 1])
            q -= profits[table_parties[j - 1]]

    return opt_set, opt_value


@njit
def _knapsack_jit_float(
        weights: np.array,
        profits: np.array,
        capacity: float,
        upper_bound: int) -> Tuple[List[int], int]:
    # This function will be replaced at runtime
    pass


# Replace the _knapsack_jit_float function with a real implementation
# It is identical to _knapsack_impl, but with the njit decorator and different types of the parameters.
exec("@njit\n" +
     inspect.getsource(_knapsack_impl)
     .replace("_knapsack_impl", "_knapsack_jit_float")
     .replace("weights: List[Union[Fraction, float, int]]", "weights: np.array")
     .replace("profits: List[int]", "profits: np.array")
     .replace("capacity: Union[Fraction, float, int]", "capacity: float"))


@njit
def _knapsack_jit_int(
        weights: np.array,
        profits: np.array,
        capacity: np.int64,
        upper_bound: int) -> Tuple[List[int], int]:
    # This function will be replaced at runtime
    pass


# Replace the _knapsack_jit_int function with a real implementation
# It is identical to _knapsack_impl, but with the njit decorator and different types of the parameters.
exec("@njit\n" +
     inspect.getsource(_knapsack_impl)
     .replace("_knapsack_impl", "_knapsack_jit_int")
     .replace("weights: List[Union[Fraction, float, int]]", "weights: np.array")
     .replace("profits: List[int]", "profits: np.array")
     .replace("capacity: Union[Fraction, float, int]", "capacity: int"))


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

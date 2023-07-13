from fractions import Fraction
from typing import List, Union, Tuple


def knapsack(weights: List[Union[Fraction, float]], profits: List[int], capacity: Union[Fraction, float],
             upper_bound: int) -> Tuple[List[int], int]:
    """
    Return the optimal set of items for the given knapsack problem.

    Takes time O(len(weights) * upper_bound).
    """
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


def sanity_knapsack(
        w: List[Union[Fraction, float]],
        p: List[int],
        c: Union[Fraction, float],
        u: int
) -> int:
    """ Solve knapsack using dynamic programming by profits. """
    n = len(w)
    assert len(p) == n

    y = [0] + [sum(w) for _ in range(1, u + 1)]

    # Fill the table
    for j in range(n):
        for q in range(u, p[j] - 1, -1):
            if y[q - p[j]] + w[j] < y[q]:
                y[q] = y[q - p[j]] + w[j]

    # Solution is the maximum index of y that does not surpass capacity
    return max([q for q in range(u + 1) if y[q] <= c])

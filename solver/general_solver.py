import logging
import math
from enum import Enum
from typing import Tuple


class Rounding(Enum):
    """Represents the rounding method used by the solver."""

    FLOOR = "floor"
    """round down the weights of the parties"""

    CEIL = "ceil"
    """round up the weights of the parties"""

    BEST_BOUND = "best_bound"
    """round with the scheme that gives the best theoretical upper bound on the solution"""

    ALL = "both"
    """round with all rounding schemes and return the best solution"""


class Params:
    def __init__(self, binary_search: bool, knapsack_binary_search: bool, linear_search: bool,
                 pruning: bool, rounding: Rounding, binary_search_iterations: int, no_jit: bool,
                 gas_limit: int, soft_memory_limit: int):
        self.linear_search = linear_search
        self.knapsack_binary_search = knapsack_binary_search or self.linear_search
        self.binary_search = binary_search or self.knapsack_binary_search
        self.pruning = pruning
        self.binary_search_iterations = binary_search_iterations
        self.rounding = rounding
        self.no_jit = no_jit
        self.gas_limit = gas_limit
        self.soft_memory_limit = soft_memory_limit


class Status(Enum):
    """Represents the type of the solution returned by a solver."""

    OPTIMAL = 1
    """returned if the solver found an optimal solution within the timeout"""

    VALID = 2
    """returned if the solver found a valid albeit non-optimal solution within the timeout"""

    NONE = 3
    """returned if the solver did not find any solution within the timeout"""

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


class Solution:
    def __init__(self, values):
        self.values = values
        self.sum = sum(values)

    def __str__(self):
        return f"Solution < sum={self.sum}, values=[{' '.join(map(str, self.values))}] >"

    def __repr__(self):
        return str(self)


def sorting_gas_cost(n: int):
    """
    Return the gas price for sorting n elements.
    :param n:
    """
    return int(math.ceil(n * math.log2(n)))


def knapsack_gas_cost(sol: Solution, upper_bound: int, return_set: bool) -> int:
    """
    Return the gas price for solving the knapsack problem.
    """
    return len(sol.values) * (upper_bound + 1) * (1 + int(return_set))


def antiknapsack_gas_cost(sol: Solution, lower_bound: int, return_set: bool) -> int:
    """
    Return the gas price for solving the antiknapsack problem.
    """
    return knapsack_gas_cost(sol, sol.sum - lower_bound, return_set)


def knapsack_memory_size(sol: Solution, upper_bound: int, return_set: bool) -> int:
    """
    Return the amount of memory to be allocated by the knapsack solver on the given parameters, in bytes.
    """
    if return_set:
        # The solver allocates a table of booleans of size n * (upper_bound + 2) to recover the solution set.
        return len(sol.values) * (upper_bound + 2)
    else:
        # When not returning the set of items in the knapsack solution,
        # the solver only allocates the amount of memory proportional to the input size, which we neglect.
        return 0


def antiknapsack_memory_size(sol: Solution, lower_bound: int, return_set: bool) -> int:
    return knapsack_memory_size(sol, sol.sum - lower_bound, return_set)


def general_solver(weights, params: Params, x_low, no_jit,
                   solution, fast_solution_check, exact_solution_check,
                   pruning_memory_requirements, pruning_gas_cost, prune) -> Tuple[Status, Solution, int]:
    gas_budget = params.gas_limit

    def charge(gas):
        nonlocal gas_budget
        assert gas <= gas_budget
        gas_budget -= gas
        logging.debug("Charging %s gas. Remaining: %s", gas, gas_budget)

    # Fast binary search with fast solution check to get an estimate of the optimal x
    if params.binary_search:
        x_high = max(weights)

        for _ in range(params.binary_search_iterations):  # TODO: come up with a good stopping condition
            x_mid = (x_high + x_low) / 2
            sol_mid = solution(x_mid)

            check_result = fast_solution_check(sol_mid, gas_budget)
            if check_result is None:
                break  # ran out of gas
            valid, gas_used = check_result

            charge(gas_used)
            if valid:
                x_low = x_mid
            else:
                x_high = x_mid

    sol_best = solution(x_low)

    # All future operations should try to leave gas for pruning if it is enabled.
    # The pruning cost will change as we improve `sol_best`.
    def get_budget():
        budget = gas_budget
        if params.pruning and pruning_memory_requirements(sol_best) <= params.soft_memory_limit:
            # When possible, leave enough gas for pruning
            pruning_cost = pruning_gas_cost(sol_best)
            if pruning_cost <= gas_budget:
                budget -= pruning_cost
        return budget

    # Refine the search using binary search with exact solution check
    if params.knapsack_binary_search:
        x_high = max(weights)

        for _ in range(params.binary_search_iterations):  # TODO: come up with a good stopping condition
            x_mid = (x_high + x_low) / 2
            sol_mid = solution(x_mid)

            check_result = exact_solution_check(sol_mid, get_budget())
            if check_result is None:
                break  # ran out of gas
            valid, gas_used = check_result

            charge(gas_used)
            if valid:
                x_low = x_mid
                sol_best = sol_mid
            else:
                x_high = x_mid

    # TODO: implement linear search

    if params.pruning:
        pruning_result = prune(sol_best, no_jit)
        if pruning_result is not None:
            sol_best, gas_used = pruning_result
            charge(gas_used)

    return Status.VALID, sol_best, params.gas_limit - gas_budget

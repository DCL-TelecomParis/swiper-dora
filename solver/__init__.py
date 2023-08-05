import logging
import math
from enum import Enum
from math import floor, ceil
from typing import Tuple, List, Union

from solver.knapsack import knapsack, knapsack_upper_bound
from solver.wq import WeightQualification
from solver.wr import WeightRestriction


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


def sorting_gas_cost(n: int):
    """
    Return the gas price for sorting n elements.
    :param n:
    """
    return int(ceil(n * math.log2(n)))


def knapsack_gas_cost(n: int, upper_bound: int, return_set: bool) -> int:
    """
    Return the gas price for solving the knapsack problem.
    """
    return n * (upper_bound + 1) * (1 + int(return_set))


def knapsack_memory_size(n: int, upper_bound: int, return_set: bool) -> int:
    """
    Return the amount of memory to be allocated by the knapsack solver on the given parameters, in bytes.
    """
    if return_set:
        # The solver allocates a table of booleans of size n * (upper_bound + 2) to recover the solution set.
        return n * (upper_bound + 2)
    else:
        # When not returning the set of items in the knapsack solution,
        # the solver only allocates the amount of memory proportional to the input size, which we neglect.
        return 0


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


def solve(inst: Union[WeightRestriction, WeightQualification], params: Params) -> Tuple[Status, Solution, int]:
    """
    Solve the Weight Restriction or Weight Qualification problem.
    Returns the status of the solution, the solution itself and the gas expended.
    """

    if isinstance(inst, WeightQualification):
        return dora(inst, params)
    elif isinstance(inst, WeightRestriction):
        return swiper(inst, params)
    else:
        raise ValueError(f"Unknown instance type {type(inst)}")


def dora(inst: WeightQualification, params: Params) -> Tuple[Status, Solution, int]:
    """
    Solve the Weight Qualification problem.
    Returns the status of the solution, the solution itself and the gas expended.
    """

    return swiper(inst.to_wr(), params)


def swiper(inst: WeightRestriction, params: Params) -> Tuple[Status, Solution, int]:
    """
    Solve the Weight Restriction problem.
    Returns the status of the solution, the solution itself and the gas expended.
    """

    assert 1 > inst.tn > inst.tw > 0

    if params.rounding == Rounding.FLOOR:
        return _swiper_floor(inst, params)
    elif params.rounding == Rounding.CEIL:
        return _swiper_ceil(inst, params)
    elif params.rounding == Rounding.BEST_BOUND:
        if inst.tw < 0.5:
            return _swiper_floor(inst, params)
        else:
            return _swiper_ceil(inst, params)
    elif params.rounding == Rounding.ALL:
        res_floor = _swiper_floor(inst, params)
        res_ceil = _swiper_ceil(inst, params)
        if res_floor[1].sum < res_ceil[1].sum:
            return res_floor
        else:
            return res_ceil
    else:
        raise ValueError(f"Unknown rounding method {params.rounding}")


def _swiper_floor(inst: WeightRestriction, params: Params) -> Tuple[Status, Solution, int]:
    return _swiper_impl(inst, params, floor, inst.total_weight / inst.n * (inst.tn - inst.tw) / inst.tn,
                        params.no_jit)


def _swiper_ceil(inst: WeightRestriction, params: Params) -> Tuple[Status, Solution, int]:
    return _swiper_impl(inst, params, ceil, inst.total_weight / inst.n * (inst.tn - inst.tw) / (1 - inst.tn),
                        params.no_jit)


def _swiper_impl(inst: WeightRestriction, params: Params, rounding, x_low, no_jit) -> Tuple[Status, Solution, int]:
    def solution(x):
        return Solution([int(rounding(w / x)) for w in inst.weights])

    def fast_solution_check(sol, gas_budget):
        gas_cost = sorting_gas_cost(inst.n)
        if gas_budget < gas_cost:
            return None
        valid = knapsack_upper_bound(inst.weights, sol.values, inst.threshold_weight) < inst.tn * sol.sum
        return valid, gas_cost

    def exact_solution_check(sol, gas_budget):
        # This is the max value up to which care to solve knapsack.
        upper_bound = floor(sol.sum * inst.tn) + 1

        memory_size = knapsack_memory_size(inst.n, upper_bound, return_set=False)
        if memory_size > params.soft_memory_limit:
            return None

        gas_cost = knapsack_gas_cost(inst.n, upper_bound, return_set=False)
        if gas_cost > gas_budget:
            return None

        profit = knapsack(inst.weights, sol.values, inst.threshold_weight, upper_bound,
                          return_set=False, no_jit=params.no_jit)
        return profit < inst.tn * sol.sum, gas_cost

    def pruning_memory_requirements(sol):
        upper_bound = floor(sol.sum * inst.tn) + 1
        return knapsack_memory_size(inst.n, upper_bound, return_set=True)

    def pruning_gas_cost(sol):
        upper_bound = floor(sol.sum * inst.tn) + 1
        return knapsack_gas_cost(inst.n, upper_bound, return_set=True)

    def prune(sol, gas_budget):
        if pruning_memory_requirements(sol) > params.soft_memory_limit:
            return None
        if pruning_gas_cost(sol) > gas_budget:
            return None
        return _swiper_prune(inst, sol, params.no_jit), pruning_gas_cost(sol)

    return _solver_impl(inst.weights, params, x_low, no_jit,
                        solution, fast_solution_check, exact_solution_check,
                        pruning_memory_requirements, pruning_gas_cost, prune)


def _solver_impl(weights, params: Params, x_low, no_jit,
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


def _swiper_prune(inst: WeightRestriction, solution: List[int], no_jit: bool = False) -> List[int]:
    """Return a solution corresponding to the pruned version the input list."""

    # The index of the party with the maximum weight
    i_max = max(range(inst.n), key=lambda i: inst.weights[i])
    solution = list(solution)

    if sum(solution) == 0 or inst.weights[i_max] > inst.threshold_weight:
        return [1 if i == i_max else 0 for i in range(inst.n)]

    best_threshold_set, best_threshold_set_t = knapsack(inst.weights, solution, inst.threshold_weight,
                                                        upper_bound=floor(sum(solution) * inst.tn) + 1,
                                                        no_jit=no_jit, return_set=True)

    if best_threshold_set_t >= inst.tn * sum(solution):
        raise Exception("Solution is not valid")
    assert best_threshold_set

    # If i is in best_threshold_set set, then it gets the number of tickets as in the original solution,
    # otherwise, we initialize it with 0
    pruned = [solution[i] if i in best_threshold_set else 0 for i in range(inst.n)]

    # We distribute tickets to the other users until the total number is ceil(best_threshold_set_t / inst.tn),
    # making sure that each user gets at most the number of tickets it got in the solution
    while sum(pruned) < smallest_greater_integer(best_threshold_set_t / inst.tn):
        # recipients are the users that have fewer tickets than the solution and are not in best_threshold_set
        recipients = [i for i in range(inst.n) if pruned[i] < solution[i] and i not in best_threshold_set]

        tickets_to_distribute = smallest_greater_integer(best_threshold_set_t / inst.tn) - sum(pruned)

        if tickets_to_distribute >= len(recipients):
            for i in recipients:
                pruned[i] += 1
        else:
            recipients.sort(key=lambda i: inst.weights[i], reverse=True)
            for i in recipients[:tickets_to_distribute]:
                pruned[i] += 1

    return pruned


def smallest_greater_integer(x) -> int:
    """Return the smallest integer greater than x."""
    return ceil(x) if x % 1 != 0 else int(x) + 1

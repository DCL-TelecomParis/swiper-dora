import math
from math import floor
from typing import List, Tuple

from solver.wr import WeightRestriction
from solver.general_solver import general_solver, Params, Status, Solution, Rounding, sorting_gas_cost, \
    knapsack_memory_size, knapsack_gas_cost
from solver.knapsack import knapsack, knapsack_upper_bound
from solver.util import smallest_greater_integer


def wr_solver(inst: WeightRestriction, params: Params) -> Tuple[Status, Solution, int]:
    """
    Solve the Weight Restriction problem.
    Returns the status of the solution, the solution itself and the gas expended.
    """

    assert 1 > inst.tn > inst.tw > 0

    if params.rounding == Rounding.FLOOR:
        return _wr_solver_floor(inst, params)
    elif params.rounding == Rounding.CEIL:
        return _wr_solver_ceil(inst, params)
    elif params.rounding == Rounding.BEST_BOUND:
        if inst.tw < 0.5:
            return _wr_solver_floor(inst, params)
        else:
            return _wr_solver_ceil(inst, params)
    elif params.rounding == Rounding.ALL:
        res_floor = _wr_solver_floor(inst, params)
        res_ceil = _wr_solver_ceil(inst, params)
        if res_floor[1].sum < res_ceil[1].sum:
            return res_floor
        else:
            return res_ceil
    else:
        raise ValueError(f"Unknown rounding method {params.rounding}")


def _wr_solver_floor(inst: WeightRestriction, params: Params) -> Tuple[Status, Solution, int]:
    return _wr_solver_impl(inst, params, floor, inst.total_weight / inst.n * (inst.tn - inst.tw) / inst.tn,
                           params.no_jit)


def _wr_solver_ceil(inst: WeightRestriction, params: Params) -> Tuple[Status, Solution, int]:
    return _wr_solver_impl(inst, params, math.ceil, inst.total_weight / inst.n * (inst.tn - inst.tw) / (1 - inst.tn),
                           params.no_jit)


def _wr_solver_impl(inst: WeightRestriction, params: Params, rounding, x_low, no_jit) -> Tuple[Status, Solution, int]:
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

        memory_size = knapsack_memory_size(sol, upper_bound, return_set=False)
        if memory_size > params.soft_memory_limit:
            return None

        gas_cost = knapsack_gas_cost(sol, upper_bound, return_set=False)
        if gas_cost > gas_budget:
            return None

        _, best_profit = knapsack(inst.weights, sol.values, inst.threshold_weight, upper_bound,
                                  return_set=False, no_jit=params.no_jit)
        return best_profit < inst.tn * sol.sum, gas_cost

    def pruning_memory_requirements(sol):
        upper_bound = floor(sol.sum * inst.tn) + 1
        return knapsack_memory_size(sol, upper_bound, return_set=True)

    def pruning_gas_cost(sol):
        upper_bound = floor(sol.sum * inst.tn) + 1
        return knapsack_gas_cost(sol, upper_bound, return_set=True)

    def prune(sol, gas_budget):
        if pruning_memory_requirements(sol) > params.soft_memory_limit:
            return None
        if pruning_gas_cost(sol) > gas_budget:
            return None
        return _wr_prune(inst, sol, params.no_jit), pruning_gas_cost(sol)

    return general_solver(inst.weights, params, x_low, no_jit,
                          solution, fast_solution_check, exact_solution_check,
                          pruning_memory_requirements, pruning_gas_cost, prune)


def _wr_prune(inst: WeightRestriction, solution: List[int], no_jit: bool = False) -> List[int]:
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

import math
from typing import Tuple

from solver import WeightRamp, Params, Status, Solution, antiknapsack_lower_bound
from solver.antiknapsack import antiknapsack
from solver.general_solver import sorting_gas_cost, knapsack_gas_cost, antiknapsack_gas_cost, general_solver, Rounding
from solver.knapsack import knapsack, knapsack_upper_bound


def ramp_solver(inst: WeightRamp, params: Params) -> Tuple[Status, Solution, int]:
    """
    Solve the Weight Restriction problem.
    Returns the status of the solution, the solution itself and the gas expended.
    """

    assert 0 < inst.alpha_w < inst.beta_w < 1
    return _ramp_solver_floor(inst, params)


def _ramp_solver_floor(inst: WeightRamp, params: Params) -> Tuple[Status, Solution, int]:
    return _ramp_solver_impl(inst, params, math.floor, inst.total_weight / inst.n * (inst.beta_w - inst.alpha_w),
                             params.no_jit)


def _ramp_solver_impl(inst: WeightRamp, params: Params, rounding, x_low, no_jit) -> Tuple[Status, Solution, int]:
    def solution(x):
        return Solution([int(rounding(w / x)) for w in inst.weights])

    def fast_solution_check(sol, gas_budget):
        gas_cost = sorting_gas_cost(inst.n)
        if gas_budget < gas_cost:
            return None
        alpha_upper_bound = knapsack_upper_bound(inst.weights, sol.values, inst.low_threshold_weight)
        beta_lower_bound = antiknapsack_lower_bound(inst.weights, sol.values, inst.high_threshold_weight)
        return alpha_upper_bound < beta_lower_bound, gas_cost

    def exact_solution_check(sol, gas_budget):
        total_gas_cost = 0

        bounds_gas_cost = 2 * sorting_gas_cost(inst.n)
        if bounds_gas_cost > gas_budget:
            return None, 0
        total_gas_cost += bounds_gas_cost

        alpha_upper_bound = knapsack_upper_bound(inst.weights, sol.values, inst.low_threshold_weight)
        beta_lower_bound = antiknapsack_lower_bound(inst.weights, sol.values, inst.high_threshold_weight)

        check_gas_cost = (knapsack_gas_cost(sol, alpha_upper_bound, return_set=False)
                          + antiknapsack_gas_cost(sol, beta_lower_bound, return_set=False))
        if total_gas_cost + check_gas_cost > gas_budget:
            return None, total_gas_cost
        total_gas_cost += check_gas_cost

        _, max_alpha_set_tickets = knapsack(inst.weights, sol.values, inst.low_threshold_weight, alpha_upper_bound,
                                            return_set=False, no_jit=params.no_jit)
        _, min_beta_set_tickets = antiknapsack(inst.weights, sol.values, inst.high_threshold_weight, beta_lower_bound,
                                               return_set=False, no_jit=params.no_jit)

        return max_alpha_set_tickets < min_beta_set_tickets, total_gas_cost

    def pruning_memory_requirements(_sol):
        return 0  # FIXME: no pruning yet

    def pruning_gas_cost(sol):
        return 0  # FIXME: no pruning yet

    def prune(sol, gas_budget):
        return sol, 0  # FIXME: no pruning yet

    return general_solver(inst.weights, params, x_low, no_jit,
                          solution, fast_solution_check, exact_solution_check,
                          pruning_memory_requirements, pruning_gas_cost, prune)

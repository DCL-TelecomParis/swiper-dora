from enum import Enum
from math import floor, ceil
from typing import Tuple, List, Optional, Union

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
                 knapsack_pruning: bool, rounding: Rounding, binary_search_iterations: int, no_jit: bool):
        self.linear_search = linear_search
        self.knapsack_binary_search = knapsack_binary_search or self.linear_search
        self.binary_search = binary_search or self.knapsack_binary_search
        self.knapsack_pruning = knapsack_pruning
        self.binary_search_iterations = binary_search_iterations
        self.rounding = rounding
        self.no_jit = no_jit

        # TODO: implement fast pruning
        # self.fast_pruning = fast_pruning
        # assert not (self.knapsack_pruning and self.fast_pruning), "Two different pruning methods are requested"


class Status(Enum):

    """Represents the type of the solution returned by a solver."""

    OPTIMAL = 1
    """returned if the solver found an optimal solution within the timeout"""

    VALID = 2
    """returned if the solver found a valid albeit non-optimal solution within the timeout"""

    NONE = 3
    """returned if the solver did not find any solution within the timeout"""


def solve(inst: Union[WeightRestriction,WeightQualification], params: Params) -> Tuple[Status, Optional[List[int]]]:
    """
    Solve the Weight Restriction or Weight Qualification problem.
    """

    if isinstance(inst, WeightQualification):
        return dora(inst, params)
    elif isinstance(inst, WeightRestriction):
        return swiper(inst, params)
    else:
        raise ValueError(f"Unknown instance type {type(inst)}")


def dora(inst: WeightQualification, params: Params) -> Tuple[Status, Optional[List[int]]]:
    """
    Solve the Weight Qualification problem.
    """

    return swiper(inst.to_wr(), params)


def swiper(inst: WeightRestriction, params: Params) -> Tuple[Status, Optional[List[int]]]:
    """
    Solve the Weight Restriction problem.
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
        status_floor, sol_floor = _swiper_floor(inst, params)
        status_ceil, sol_ceil = _swiper_ceil(inst, params)
        if sum(sol_floor) < sum(sol_ceil):
            return status_floor, sol_floor
        else:
            return status_ceil, sol_ceil
    else:
        raise ValueError(f"Unknown rounding method {params.rounding}")


def _swiper_floor(inst: WeightRestriction, params: Params) -> Tuple[Status, Optional[List[int]]]:
    return _swiper_impl(inst, params, floor, inst.total_weight / inst.n * (inst.tn - inst.tw) / inst.tn,
                        params.no_jit)


def _swiper_ceil(inst: WeightRestriction, params: Params) -> Tuple[Status, Optional[List[int]]]:
    return _swiper_impl(inst, params, ceil, inst.total_weight / inst.n * (inst.tn - inst.tw) / (1 - inst.tn),
                        params.no_jit)


def _swiper_impl(inst: WeightRestriction, params: Params, rnd, x_low, no_jit) -> Tuple[Status, Optional[List[int]]]:
    """
    :param rnd: rounding function
    :param x_low: lower bound on the optimal value of X
    """

    if params.binary_search:
        x_high = max(inst.weights)

        for _ in range(params.binary_search_iterations):  # TODO: come up with a good stopping condition
            x_mid = (x_high + x_low) / 2
            t_mid = [rnd(inst.weights[i] / x_mid) for i in range(inst.n)]

            if knapsack_upper_bound(inst.weights, t_mid, inst.threshold_weight) < inst.tn * sum(t_mid):
                x_low = x_mid
            else:
                x_high = x_mid

    # Refine the search using knapsack and binary search
    if params.knapsack_binary_search:
        x_high = max(inst.weights)

        for _ in range(params.binary_search_iterations):  # TODO: come up with a good stopping condition
            x_mid = (x_high + x_low) / 2
            t_mid = [rnd(inst.weights[i] / x_mid) for i in range(inst.n)]
            sum_t_x_mid = sum(t_mid)

            best_threshold_set, best_threshold_set_t = knapsack(inst.weights, t_mid, inst.threshold_weight,
                                                                upper_bound=floor(sum_t_x_mid * inst.tn) + 1,
                                                                no_jit=no_jit)

            if best_threshold_set_t < inst.tn * sum_t_x_mid:
                x_low = x_mid
            else:
                x_high = x_mid

    # The best solution found so far
    t_best = [rnd(inst.weights[i] / x_low) for i in range(inst.n)]

    # Finish the search going through the rest of relevant values of X one by one
    if params.linear_search:
        t_prime = t_best.copy()
        while True:
            sum_t_prime = sum(t_prime)
            holders = [i for i in range(inst.n) if t_prime[i] > 0]
            best_threshold_set, best_threshold_set_t = knapsack(inst.weights, t_prime, inst.threshold_weight,
                                                                upper_bound=floor(sum_t_prime * inst.tn) + 1,
                                                                no_jit=no_jit)
            if best_threshold_set == [] or sum_t_prime == 0:
                break

            if best_threshold_set_t <= inst.tn * sum_t_prime:
                t_best = t_prime.copy()

            x_prime = min([inst.weights[i] / t_prime[i] for i in best_threshold_set])
            for i in holders:
                if inst.weights[i] / t_prime[i] == x_prime:
                    t_prime[i] -= 1
                else:
                    t_prime[i] = rnd(inst.weights[i] / x_prime)

    if params.knapsack_pruning:
        t_best = prune(inst, t_best, no_jit)

    return Status.VALID, t_best


def prune(inst: WeightRestriction, solution: List[int], no_jit: bool = False) -> List[int]:
    """Return a solution corresponding to the pruned version the input list."""

    # The index of the party with the maximum weight
    i_max = max(range(inst.n), key=lambda i: inst.weights[i])
    solution = list(solution)

    if sum(solution) == 0 or inst.weights[i_max] > inst.threshold_weight:
        return [1 if i == i_max else 0 for i in range(inst.n)]

    best_threshold_set, best_threshold_set_t = knapsack(inst.weights, solution, inst.threshold_weight,
                                                        upper_bound=floor(sum(solution) * inst.tn) + 1,
                                                        no_jit=no_jit)

    if best_threshold_set_t > inst.tn * sum(solution):
        raise Exception("Solution is not valid")
    assert best_threshold_set

    # If i is in best_threshold_set set, then it gets the number of tickets as in the original solution,
    # otherwise, we initialize it with 0
    pruned = [solution[i] if i in best_threshold_set else 0 for i in range(inst.n)]

    # We distribute tickets to the other users until the total number is ceil(best_threshold_set_t / inst.tn),
    # making sure that each user gets at most the number of tickets it got in the solution
    while sum(pruned) < ceil(best_threshold_set_t / inst.tn):
        # recipients are the users that have fewer tickets than the solution and are not in best_threshold_set
        recipients = [i for i in range(inst.n) if pruned[i] < solution[i] and i not in best_threshold_set]

        tickets_to_distribute = ceil(best_threshold_set_t / inst.tn) - sum(pruned)

        if tickets_to_distribute >= len(recipients):
            for i in recipients:
                pruned[i] += 1
        else:
            recipients.sort(key=lambda i: inst.weights[i], reverse=True)
            for i in recipients[:tickets_to_distribute]:
                pruned[i] += 1

    return pruned

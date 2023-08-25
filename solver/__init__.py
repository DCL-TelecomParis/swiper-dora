from typing import Tuple, Union

from solver.antiknapsack import antiknapsack_lower_bound
from solver.general_solver import Params, Status, Solution
from solver.knapsack import knapsack, knapsack_upper_bound
from solver.ramp import WeightRamp
from solver.ramp_solver import ramp_solver
from solver.wq import WeightQualification
from solver.wq_solver import wq_solver
from solver.wr import WeightRestriction
from solver.wr_solver import wr_solver


def solve(inst: Union[WeightRestriction, WeightQualification], params: Params) -> Tuple[Status, Solution, int]:
    """
    Solve the Weight Restriction or Weight Qualification problem.
    Returns the status of the solution, the solution itself and the gas expended.
    """

    if isinstance(inst, WeightQualification):
        return wq_solver(inst, params)
    elif isinstance(inst, WeightRestriction):
        return wr_solver(inst, params)
    elif isinstance(inst, WeightRamp):
        return ramp_solver(inst, params)
    else:
        raise ValueError(f"Unknown instance type {type(inst)}")

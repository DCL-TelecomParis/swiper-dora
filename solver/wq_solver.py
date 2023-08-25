from typing import Tuple

from solver.wq import WeightQualification
from solver.general_solver import Params, Status, Solution
from solver.wr_solver import wr_solver


def wq_solver(inst: WeightQualification, params: Params) -> Tuple[Status, Solution, int]:
    """
    Solve the Weight Qualification problem.
    Returns the status of the solution, the solution itself and the gas expended.
    """

    return wr_solver(inst.to_wr(), params)

#!/usr/bin/env python3

import argparse
import logging
import math
import sys
from fractions import Fraction
from typing import List

from solver import solve, Params, Status, Rounding
from solver.knapsack import MAX_INT_64, overflow_error
from solver.wr import WeightRestriction
from solver.wq import WeightQualification

# from kap import kap_solve
# from kap_instance import ProblemInstance, Status
# from utils import str_to_fracfloat

logger = logging.getLogger(__name__)


def parse_input(inp: str) -> List[Fraction]:
    return [Fraction(s) for s in inp.split()]


def main(argv: List[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="wrp",
        description="Solver to the Weight Restriction (WR) and Weight Qualification (WQ) problems, "
                    "as defined in the Swiper and Dora paper.",
    )

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("-f", "--float", action="store_true",
                               help="Use floating point numbers instead of exact representations of rational numbers. "
                                    "This implementation may occasionally produce incorrect results or even crash due "
                                    "to the rounding errors. It is also not guaranteed to produce the same results "
                                    "on different platforms.")
    common_parser.add_argument("--no-jit", action="store_true",
                               help="Do not use JIT compilation. This guarantees that there are no integer overflows, "
                                    "but may result in a significant performance degradation.")
    common_parser.add_argument("-v", "--verbose", action="store_true", default=False,
                               help="Set this flag to enable verbose logging.")
    common_parser.add_argument("--speed", type=int, default=5, choices=range(1, 11), metavar="[1-10]",
                               help="Set the speed of the solver.\n"
                                    # "0: Find exact solution (requires exponential time and memory).\n"
                                    "1: Most precise polynomial approximate algorithm.\n"
                                    "3: Recommended for small inputs (n < 1000).\n"
                                    "5: Recommended for most inputs.\n"
                                    "7: Maximum value at which pruning is performed.\n"
                                    "10: Fastest, linear-time approximate algorithm, without pruning.")
    common_parser.add_argument("input_file", type=argparse.FileType("r"), default=sys.stdin, nargs='?',
                               help="The path to the input file. "
                                    "If absent, the standard input will be used. "
                                    "The input consists of weights of parties separated by any whitespaces. "
                                    "Rational numbers are accepted.")

    subparsers = parser.add_subparsers(title="solver", required=True, dest="solver")

    swiper_aliases = ["wp"]
    wr_parser = subparsers.add_parser("swiper", aliases=swiper_aliases, parents=[common_parser],
                                      help="Solve the Weight Restriction problem, i.e., "
                                           "ensure that any group of parties with less than "
                                           "alpha_w fraction of total weight obtains less than "
                                           "alpha_n fraction of total tickets.")
    wr_parser.add_argument("--tw", "--alpha_w", type=Fraction, required=True,
                           help="The weighted threshold. Corresponds to alpha_w in the paper. "
                                "Must be smaller than the nominal threshold alpha_n.")
    wr_parser.add_argument("--tn", "--alpha_n", type=Fraction, required=True,
                           help="The nominal threshold. Corresponds to alpha_n in the paper. "
                                "Must be greater than the weighted threshold alpha_w.")

    dora_aliases = ["wq"]
    wq_parser = subparsers.add_parser("dora", aliases=dora_aliases, parents=[common_parser],
                                      help="Solve the Weight Qualification problem, i.e., "
                                           "ensure that any group of parties with more than "
                                           "beta_w fraction of total weight obtains more than "
                                           "beta_n fraction of total tickets.")
    wq_parser.add_argument("--tw", "--beta_w", type=Fraction, required=True,
                           help="The weighted threshold. Corresponds to beta_w in the paper. "
                                "Must be greater than the nominal threshold beta_n.")
    wq_parser.add_argument("--tn", "--beta_n", type=Fraction, required=True,
                           help="The nominal threshold. Corresponds to beta_n in the paper. "
                                "Must be smaller than the weighted threshold beta_w.")

    args = parser.parse_args(argv)

    # Set up logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s")

    weights = parse_input(args.input_file.read())

    if args.float:
        args.tw = float(args.tw)
        args.tn = float(args.tn)
        weights = [float(w) for w in weights]
    else:
        # Convert weights to integers
        lcm = 1
        for w in weights + [Fraction(args.tw), Fraction(args.tn)]:
            lcm = lcm * w.denominator // math.gcd(lcm, w.denominator)
        weights = [int(w * lcm) for w in weights]

    if args.solver in swiper_aliases:
        args.solver = "swiper"
    elif args.solver in dora_aliases:
        args.solver = "dora"

    if args.solver == "swiper":
        inst = WeightRestriction(weights, args.tw, args.tn)
    else:
        inst = WeightQualification(weights, args.tw, args.tn)

    logger.info("Problem: %s", inst)
    logger.info("Total weight: %s", inst.total_weight)
    logger.info("Threshold weight: %s", inst.threshold_weight)

    floor_status, floor_solution = solve(inst, Params(
        binary_search=args.speed <= 9,
        knapsack_pruning=args.speed <= 7,
        knapsack_binary_search=args.speed <= 5,
        linear_search=args.speed <= 4,
        binary_search_iterations=30,
        rounding=Rounding.FLOOR,
        no_jit=args.no_jit,
    ))

    ceil_status, ceil_solution = solve(inst, Params(
        binary_search=args.speed <= 6,
        # Knapsack can be very slow for the ceiling distribution due to a bad starting distribution.
        knapsack_pruning=args.speed <= 5,
        knapsack_binary_search=args.speed <= 3,
        linear_search=args.speed <= 1,
        binary_search_iterations=30,
        rounding=Rounding.CEIL,
        no_jit=args.no_jit,
    ))

    status, solution = ((floor_status, floor_solution) if sum(floor_solution) < sum(ceil_solution)
                        else (ceil_status, ceil_solution))

    if status == Status.NONE:
        print("No solution found within the time limit.")
        return
    if status == Status.OPTIMAL:
        print("Optimal solution found:")
    elif status == Status.VALID:
        print("Valid solution found:")
    else:
        assert False

    assert solution is not None
    logger.info(solution)
    print(f"Total tickets allocated: {sum(solution)}.")


if __name__ == '__main__':
    main(sys.argv[1:])

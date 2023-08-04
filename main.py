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


def parse_gas_limit(inp: str) -> int:
    if inp[-1] in ["M", "m", "B", "b"]:
        suffix = inp[-1]
        val = Fraction(inp[:-1])
    else:
        suffix = ""
        val = Fraction(inp)

    if suffix in ["M", "m"]:
        val *= 10 ** 6
    elif suffix in ["B", "b"]:
        val *= 10 ** 9

    return int(val)


def parse_soft_memory_limit(inp: str) -> int:
    if inp[-1] in ["K", "k", "M", "m", "G", "g"]:
        suffix = inp[-1]
        val = Fraction(inp[:-1])
    else:
        suffix = ""
        val = Fraction(inp)

    if suffix in ["K", "k"]:
        val *= 10 ** 3
    elif suffix in ["M", "m"]:
        val *= 10 ** 6
    elif suffix in ["G", "g"]:
        val *= 10 ** 9

    return int(val)


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
    common_parser.add_argument("-vv", "--very-verbose", action="store_true", default=False,
                               help="Set this flag to enable very verbose logging.")
    common_parser.add_argument("--speed", type=int, default=5, choices=range(1, 11), metavar="[1-10]",
                               help="Set the speed of the solver.\n"
                                    # "0: Find exact solution (requires exponential time and memory).\n"
                                    "1: Most precise polynomial approximate algorithm.\n"
                                    "3: Recommended for small inputs (n < 1000).\n"
                                    "5: Recommended for medium-size inputs (n < 50'000). "
                                    "O(n^2 log n / |tn - tw|) time complexity.\n"
                                    "7: Maximum value at which pruning is performed. "
                                    "O(n^2 / |tn - tw|) time complexity.\n"
                                    "10: Fastest, linear-time approximate algorithm, without pruning. "
                                    "O(n) time complexity.")
    common_parser.add_argument("-l", "--gas-limit", type=str, default=None, metavar="GAS",
                               help="A way to express the time limit in a deterministic way. "
                                    "1 unit of GAS roughly corresponds to 1 simple memory access. "
                                    "1 billion GAS roughly corresponds to 1 second of computation. "
                                    "For convenience, suffixes M (1e6) and B (1e9) are supported. "
                                    "Examples: 1.5e9, 10B, 1M.")
    common_parser.add_argument("--soft-memory-limit", type=str, default=None, metavar="BYTES",
                               help="Specifies the maximum amount of memory allocated by the knapsack solver in bytes. "
                                    "When a knapsack solver instance may exceed this limit, it will not be called. "
                                    "This decision is deterministic and platform-independent. "
                                    "The actual memory utilization may slightly exceed this number and "
                                    "depends on the Python garbage collector. "
                                    "For convenience, suffixes K (1e3), M (1e6), G (1e9) are supported. "
                                    "Examples: 1.5e9, 10G, 1M.")
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
                                "Must be smaller than the nominal threshold alpha_n. "
                                "Can be fractional (e.g., 0.01 or 5/7).")
    wr_parser.add_argument("--tn", "--alpha_n", type=Fraction, required=True,
                           help="The nominal threshold. Corresponds to alpha_n in the paper. "
                                "Must be greater than the weighted threshold alpha_w. "
                                "Can be fractional (e.g., 0.01 or 5/7).")

    dora_aliases = ["wq"]
    wq_parser = subparsers.add_parser("dora", aliases=dora_aliases, parents=[common_parser],
                                      help="Solve the Weight Qualification problem, i.e., "
                                           "ensure that any group of parties with more than "
                                           "beta_w fraction of total weight obtains more than "
                                           "beta_n fraction of total tickets.")
    wq_parser.add_argument("--tw", "--beta_w", type=Fraction, required=True,
                           help="The weighted threshold. Corresponds to beta_w in the paper. "
                                "Must be greater than the nominal threshold beta_n. "
                                "Can be fractional (e.g., 0.01 or 5/7).")
    wq_parser.add_argument("--tn", "--beta_n", type=Fraction, required=True,
                           help="The nominal threshold. Corresponds to beta_n in the paper. "
                                "Must be smaller than the weighted threshold beta_w. "
                                "Can be fractional (e.g., 0.01 or 5/7).")

    args = parser.parse_args(argv)

    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if args.very_verbose else logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s")

    # Disable numba debug logging
    if args.very_verbose and not args.no_jit:
        numba_logger = logging.getLogger('numba')
        numba_logger.setLevel(logging.INFO)

    weights = parse_input(args.input_file.read())

    # Help the IDE determine the types
    args.tw = Fraction(args.tw)
    args.tn = Fraction(args.tn)

    if args.float:
        args.tw = float(args.tw)
        args.tn = float(args.tn)
        weights = [float(w) for w in weights]
    else:
        # Convert weights to integers
        denominator_lcm = lcm(w.denominator for w in weights + [args.tw, args.tn])
        numerator_gcd = gcd(w.numerator for w in weights)
        weights = [int(w * denominator_lcm // numerator_gcd) for w in weights]

    if args.solver in swiper_aliases:
        args.solver = "swiper"
    elif args.solver in dora_aliases:
        args.solver = "dora"

    if args.solver == "swiper":
        inst = WeightRestriction(weights, args.tw, args.tn)
    else:
        inst = WeightQualification(weights, args.tw, args.tn)

    if args.gas_limit is not None:
        gas_limit = parse_gas_limit(args.gas_limit)
    else:
        gas_limit = parse_gas_limit("1e18")

    if args.soft_memory_limit is not None:
        soft_memory_limit = parse_soft_memory_limit(args.soft_memory_limit)
    else:
        soft_memory_limit = parse_soft_memory_limit("1000G")

    logger.info("Problem: %s", inst)
    logger.info("Total weight: %s", inst.total_weight)
    logger.info("Threshold weight: %s", inst.threshold_weight)
    logger.info("Gas limit: %s", gas_limit)
    logger.info("Soft memory limit: %s", soft_memory_limit)

    ceil_status, ceil_solution, ceil_gas_usage = solve(inst, Params(
        binary_search=args.speed <= 6,
        knapsack_pruning=args.speed <= 4,
        knapsack_binary_search=args.speed <= 2,
        linear_search=args.speed <= 1,
        binary_search_iterations=30,
        rounding=Rounding.CEIL,
        no_jit=args.no_jit,
        gas_limit=gas_limit // 2,
        soft_memory_limit=soft_memory_limit,
    ))
    logger.info("Ceiling solution: %s", sum(ceil_solution))
    logger.info("Gas expanded by the ceiling solution: %s", ceil_gas_usage)

    floor_status, floor_solution, floor_gas_usage = solve(inst, Params(
        binary_search=args.speed <= 9,
        knapsack_pruning=args.speed <= 7,
        knapsack_binary_search=args.speed <= 5,
        linear_search=args.speed <= 3,
        binary_search_iterations=30,
        rounding=Rounding.FLOOR,
        no_jit=args.no_jit,
        gas_limit=gas_limit - ceil_gas_usage,
        soft_memory_limit=soft_memory_limit,
    ))
    logger.info("Floor solution: %s", sum(floor_solution))
    logger.info("Gas expanded by the floor solution: %s", floor_gas_usage)

    status, solution = ((floor_status, floor_solution) if sum(floor_solution) < sum(ceil_solution)
                        else (ceil_status, ceil_solution))

    if status == Status.NONE:
        print("No solution found within the time limit.")
        return
    if status == Status.OPTIMAL:
        print("Optimal solution found.")
    elif status == Status.VALID:
        print("Valid solution found.")
    else:
        assert False

    assert solution is not None
    logger.info(solution)
    print(f"Total tickets allocated: {sum(solution)}.")
    print(f"Total gas expanded: {floor_gas_usage + ceil_gas_usage}")


def lcm(xs):
    res = 1
    for x in xs:
        res = res * x // math.gcd(res, x)
    return res


def gcd(xs):
    res = 0
    for x in xs:
        res = math.gcd(res, x)
    return res


if __name__ == '__main__':
    main(sys.argv[1:])

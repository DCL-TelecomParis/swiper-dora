import math

from numba.core.extending import register_jitable


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


# Unfortunately, Python's native `reserved` function is not jitable.
@register_jitable
def rev(rng: range) -> range:
    return range(rng.start + (len(rng) - 1) * rng.step, rng.start - rng.step, -rng.step)


def smallest_greater_integer(x) -> int:
    """Return the smallest integer greater than x."""
    return math.ceil(x) if x % 1 != 0 else int(x) + 1

from fractions import Fraction
from typing import List, Union


class WeightRestriction:

    """Represents an instance of the Weight Restriction problem."""

    def __init__(self,
                 weights: List[Union[Fraction, float, int]],
                 tw: Union[Fraction, float],
                 tn: Union[Fraction, float]):
        """
        Create a new instance.

        :param weights: list of weights of the parties
        :param tw: weighted threshold
        :param tn: nominal threshold
        """
        # Number of parties
        self.n = len(weights)
        # List of weights of the parties
        self.weights = weights
        # Total weight of all parties
        self.total_weight = sum(weights)
        # Maximum possible total weight of Byzantine parties
        self.threshold_weight = tw * self.total_weight
        # Threshold on the fraction of total weight controlled by the adversary
        self.tw = tw
        # Threshold on the fraction of tickets allocated to the adversary
        self.tn = tn

    def __str__(self):
        return f"WeightRestriction < " \
               f"n={self.n}, weights=[{' '.join(map(str, self.weights))}], tw={self.tw}, tn={self.tn}" \
               f" >"

    def __repr__(self):
        return str(self)

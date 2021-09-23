'''Linear arm with Gaussian noise.'''

__version__ = "0.1"

from .Arm import Arm
from numpy import dot, random


class LinearArm(Arm):
    """Stochastic possibly noisy reward generator"""

    def __init__(self, regression_vector):
        self._regression_vector = regression_vector

    def __str__(self):
        return "StochasticArm. Linear regression vector {}".format(self.regression_vector)

    def draw(self, chosen_context):
        ''' The noise is added by the MAB Class. '''
        assert chosen_context.shape == self._regression_vector.shape, "Incoherent vectors dimension. LinearArm.draw()"
        return dot(self._regression_vector, chosen_context)

import abc
from typing import Tuple

import math


class Model(abc.ABC):
    # TODO: add update method

    @abc.abstractmethod
    def calculate_expected_result(self, *args) -> float:
        ...


class LogisticRegression(Model):

    def __init__(self, alpha: Tuple[float, ...], beta: float):
        self.alpha: Tuple[float, ...] = alpha
        self.beta: float = beta

    def calculate_expected_result(self, *args) -> float:
        if len(args) != len(self.alpha):
            raise ValueError("Length of args/values must match length of alpha/coefficients.")
        # I couldn't see any obvious speed-up from using NumPy/Numba data
        # structures but should revisit this.
        exp = sum(k * v for k, v in zip(self.alpha, args))
        return 1 / (1 + math.pow(10, -exp / (2 * self.beta)))

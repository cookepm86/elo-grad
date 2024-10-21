import abc
from collections import defaultdict
from typing import Tuple, Optional, Dict

import math


class Model(abc.ABC):

    loss: str

    def __init__(
        self,
        default_init_rating: float,
        init_ratings: Optional[Dict[str, Tuple[Optional[int], float]]] = None,
    ) -> None:
        self.ratings: Dict[str, Tuple[Optional[int], float]] = defaultdict(
            lambda: (None, default_init_rating)
        )
        self.init_ratings: Optional[Dict[str, Tuple[Optional[int], float]]] = init_ratings
        if self.init_ratings is not None:
            self.ratings = self.ratings | self.init_ratings

    @abc.abstractmethod
    def calculate_gradient(self, y: int, *args) -> float:
        ...

    @abc.abstractmethod
    def calculate_expected_score(self, *args) -> float:
        ...


class Optimizer(abc.ABC):

    @abc.abstractmethod
    def calculate_update_step(self, model: Model, y: int, entity_1: str, entity_2: str) -> Tuple[float, ...]:
        ...


class LogisticRegression(Model):

    loss: str = "log-loss"

    def __init__(
        self,
        beta: float,
        default_init_rating: float,
        init_ratings: Optional[Dict[str, Tuple[Optional[int], float]]],
    ) -> None:
        super().__init__(default_init_rating, init_ratings)
        self.beta: float = beta

    def calculate_gradient(self, y: int, *args) -> float:
        if y not in {0, 1}:
            raise ValueError("Invalid result value %s", y)
        y_pred: float = self.calculate_expected_score(*args)

        return y - y_pred

    def calculate_expected_score(self, *args) -> float:
        # I couldn't see any obvious speed-up from using NumPy/Numba data
        # structures but should revisit this.
        return 1 / (1 + math.pow(10, -sum(args) / (2 * self.beta)))


class SGDOptimizer(Optimizer):

    def __init__(self, k_factor: float) -> None:
        self.k_factor: float = k_factor

    def calculate_update_step(self, model: Model, y: int, entity_1: str, entity_2: str) -> Tuple[float, ...]:
        grad: float = model.calculate_gradient(
            y,
            model.ratings[entity_1][1],
            -model.ratings[entity_2][1],
        )
        step: float = self.k_factor * grad

        return step, -step

    def update_model(self, model: Model, y: int, entity_1: str, entity_2: str, t: Optional[int] = None) -> None:
        delta = self.calculate_update_step(model, y, entity_1, entity_2)
        model.ratings[entity_1] = (
            t,
            model.ratings[entity_1][1] + delta[0],
        )
        model.ratings[entity_2] = (
            t,
            model.ratings[entity_2][1] + delta[1],
        )

import abc
from collections import defaultdict
from typing import Tuple, Optional, Dict, List

import math
import pandas as pd
from array import array

__all__ = ["EloEstimator", "LogisticRegression", "SGDOptimizer"]


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

    @abc.abstractmethod
    def update_model(self, model: Model, y: int, entity_1: str, entity_2: str, t: Optional[int] = None) -> None:
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


class EloEstimator:
    """
    Elo rating system classifier.

    Attributes
    ----------
    columns : List[str]
        [entity_1, entity_2, result] columns names.
    entity_cols : Tuple[str, str]
        Names of columns identifying the names of the entities playing the games.
    model : Model
        Underlying statistical model.
    optimizer : Optimizer
        Optimizer to update the model.
    score_col : str
        Name of score column (1 if entity_1 wins and 0 if entity_2 wins).
        Draws are not currently supported.

    Methods
    -------
    transform(X)
        Calculate ratings/expected scores based on historical matches.
    """

    def __init__(
        self,
        k_factor: float,
        default_init_rating: float,
        beta: float = 200,
        init_ratings: Optional[Dict[str, Tuple[Optional[int], float]]] = None,
        entity_cols: Tuple[str, str] = ("entity_1", "entity_2"),
        score_col: str = "score",
    ) -> None:
        """
        Parameters
        ----------
        k_factor : float
            Elo K-factor/step-size for gradient descent.
        default_init_rating : float
            Default initial rating for entities.
        beta : float
            Normalization factor for ratings when computing expected score.
        init_ratings : Optional[Dict[str, Tuple[Optional[int], float]]]
            Initial ratings for entities (dictionary of form entity: (Unix timestamp, rating))
        entity_cols : Tuple[str, str]
            Names of columns identifying the names of the entities playing the games.
        score_col : str
            Name of score column (1 if entity_1 wins and 0 if entity_2 wins).
            Draws are not currently supported.
        """
        self.entity_cols: Tuple[str, str] = entity_cols
        self.score_col: str = score_col
        self.columns: List[str] = list(entity_cols) + [score_col]
        self.model: Model = LogisticRegression(
            beta=beta,
            default_init_rating=default_init_rating,
            init_ratings=init_ratings,
        )
        self.optimizer: Optimizer = SGDOptimizer(k_factor=k_factor)

    def _update_ratings(self, t: int, rating_deltas: Dict[str, float]) -> None:
        for entity in rating_deltas:
            self.model.ratings[entity] = (t, self.model.ratings[entity][1] + rating_deltas[entity])

    def transform(self, X: pd.DataFrame) -> pd.Series:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")
        X = X[self.columns]

        if not X.index.is_monotonic_increasing:
            raise ValueError("Index must be sorted.")
        current_ix: int = X.index[0]

        preds = array("f")
        rating_deltas: Dict[str, float] = defaultdict(float)
        for row in X.itertuples(index=True):
            ix, entity_1, entity_2, score = row
            if ix != current_ix:
                self._update_ratings(ix, rating_deltas)
                current_ix, rating_deltas = ix, defaultdict(float)

            expected_score: float = self.model.calculate_expected_score(
                self.model.ratings[entity_1][1],
                -self.model.ratings[entity_2][1],
            )
            preds.append(expected_score)

            rating_delta: Tuple[float, ...] = self.optimizer.calculate_update_step(
                model=self.model,
                y=score,
                entity_1=entity_1,
                entity_2=entity_2,
            )
            rating_deltas[entity_1] += rating_delta[0]
            rating_deltas[entity_2] += rating_delta[1]

        self._update_ratings(ix, rating_deltas)

        return pd.Series(preds, index=X.index, name="expected_score")

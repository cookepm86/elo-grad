import abc
from array import array
from collections import defaultdict
from typing import Tuple, Optional, Dict, List, Callable

import math
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import log_loss

from .plot import HistoryPlotterMixin

__all__ = ["EloEstimator", "LogisticRegression", "RatingSystemMixin", "SGDOptimizer"]


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


class RatingSystemMixin:
    """
    Mixin class for rating systems.

    This mixin defines the following functionality:

    - `_estimator_type` class attribute defaulting to `"rating-system"`;
    - `score` method that default to :func:`~sklearn.metrics.log_loss`.
    - enforce that `fit` does not require `y` to be passed through the `requires_y` tag.

    Read more in the :ref:`User Guide <rolling_your_own_estimator>`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.base import BaseEstimator
    >>> from elo_grad import RatingSystemMixin
    >>> # Mixin classes should always be on the left-hand side for a correct MRO
    >>> class MyEstimator(RatingSystemMixin, BaseEstimator):
    ...     def __init__(self, *, param=1):
    ...         self.param = param
    ...     def fit(self, X, y=None):
    ...         self.is_fitted_ = True
    ...         return self
    ...     def predict(self, X):
    ...         return np.full(shape=X.shape[0], fill_value=self.param)
    >>> estimator = MyEstimator(param=1)
    >>> X = np.array([[1, 2], [2, 3], [3, 4]])
    >>> y = np.array([1, 0, 1])
    >>> estimator.fit(X, y).predict(X)
    array([1, 1, 1])
    >>> estimator.score(X, y)
    0.66...
    """

    _estimator_type = "classifier"
    classes_ = [[0, 1]]

    def score(self, X, y, sample_weight=None):
        """
        Return the log-loss on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for `X`.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of ``self.predict(X)`` w.r.t. `y`.
        """
        return log_loss(
            y, self.predict_proba(X)[:, 1], sample_weight=sample_weight
        )

    def _more_tags(self):
        return {"requires_y": False}


class EloEstimator(HistoryPlotterMixin, RatingSystemMixin, BaseEstimator):
    """
    Elo rating system classifier.

    Attributes
    ----------
    beta : float
        Normalization factor for ratings when computing expected score.
    columns : List[str]
        [entity_1, entity_2, result] columns names.
    default_init_rating : float
        Default initial rating for entities.
    entity_cols : Tuple[str, str]
        Names of columns identifying the names of the entities playing the games.
    init_ratings : Optional[Dict[str, Tuple[Optional[int], float]]]
        Initial ratings for entities (dictionary of form entity: (Unix timestamp, rating))
    k_factor : float
        Elo K-factor/step-size for gradient descent.
    model : Model
        Underlying statistical model.
    optimizer : Optimizer
        Optimizer to update the model.
    rating_history : List[Tuple[Optional[int], float]]
        Historical ratings of entities (if track_rating_history is True).
    score_col : str
        Name of score column (1 if entity_1 wins and 0 if entity_2 wins).
        Draws are not currently supported.
    track_rating_history : bool
        Flag to track historical ratings of entities.

    Methods
    -------
    fit(X, y=None)
        Fit Elo rating system/calculate ratings.
    record_ratings()
        Record the current ratings of entities.
    predict_proba(X)
        Produce probability estimates.
    predict(X)
        Predict outcome of game.
    """

    def __init__(
        self,
        k_factor: float = 20,
        default_init_rating: float = 1200,
        beta: float = 200,
        init_ratings: Optional[Dict[str, Tuple[Optional[int], float]]] = None,
        entity_cols: Tuple[str, str] = ("entity_1", "entity_2"),
        score_col: str = "score",
        track_rating_history: bool = False,
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
        track_rating_history : bool
            Flag to track historical ratings of entities.
        """
        self.entity_cols: Tuple[str, str] = entity_cols
        self.score_col: str = score_col
        self.columns: List[str] = list(entity_cols) + [score_col]
        self.beta: float = beta
        self.default_init_rating: float = default_init_rating
        self.init_ratings: Optional[Dict[str, Tuple[Optional[int], float]]] = init_ratings
        self.model: Model = LogisticRegression(
            beta=beta,
            default_init_rating=default_init_rating,
            init_ratings=init_ratings,
        )
        self.k_factor: float = k_factor
        self.optimizer: Optimizer = SGDOptimizer(k_factor=k_factor)
        self.track_rating_history: bool = track_rating_history
        self.rating_history: List[Tuple[Optional[int], float]] = defaultdict(list)  # type:ignore

    @staticmethod
    def _reinitialize_rating_system(method: Callable):
        """
        Decorator to reinitialize the rating system after parameter changes.
        Helpful when peforming a grid search.

        Parameters
        ----------
        method : Callable
            Method to decorate
        """

        def wrapper(self, **params):
            result = method(self, **params)
            self.model = LogisticRegression(
                beta=self.beta,
                default_init_rating=self.default_init_rating,
                init_ratings=self.init_ratings,
            )

            return result

        return wrapper

    @_reinitialize_rating_system
    def set_params(self, **params):
        return super().set_params(**params)

    def _update_ratings(self, t: int, rating_deltas: Dict[str, float]) -> None:
        for entity in rating_deltas:
            self.model.ratings[entity] = (t, self.model.ratings[entity][1] + rating_deltas[entity])

    def record_ratings(self) -> None:
        """
        Record the current ratings of entities.
        """
        for k, v in self.model.ratings.items():
            self.rating_history[k].append(v)  # type:ignore

    def _transform(self, X: pd.DataFrame, return_expected_score: bool) -> Optional[np.ndarray]:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")
        X = X[self.columns]

        if not X.index.is_monotonic_increasing:
            raise ValueError("Index must be sorted.")
        current_ix: int = X.index[0]

        preds = array("f") if return_expected_score else None
        rating_deltas: Dict[str, float] = defaultdict(float)
        for row in X.itertuples(index=True):
            ix, entity_1, entity_2, score = row
            if ix != current_ix:
                self._update_ratings(ix, rating_deltas)
                current_ix, rating_deltas = ix, defaultdict(float)
                if self.track_rating_history:
                    self.record_ratings()

            expected_score: float = self.model.calculate_expected_score(
                self.model.ratings[entity_1][1],
                -self.model.ratings[entity_2][1],
            )
            if return_expected_score:
                preds.append(expected_score)  # type:ignore

            rating_delta: Tuple[float, ...] = self.optimizer.calculate_update_step(
                model=self.model,
                y=score,
                entity_1=entity_1,
                entity_2=entity_2,
            )
            rating_deltas[entity_1] += rating_delta[0]
            rating_deltas[entity_2] += rating_delta[1]

        self._update_ratings(ix, rating_deltas)
        if self.track_rating_history:
            self.record_ratings()

        if return_expected_score:
            return np.array(preds)
        return None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        self._transform(X, return_expected_score=False)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        preds = self._transform(X, return_expected_score=True)
        return np.vstack((1 - preds, preds)).T  # type:ignore

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(X)[:, 1] > 0.5

import pandas as pd
import pytest

from elo_grad import LogisticRegression, SGDOptimizer, EloEstimator


class TestLogisticRegression:
    def test_calculate_expected_score_equal_ratings(self):
        model = LogisticRegression(
            default_init_rating=1,
            init_ratings=None,
            beta=1,
        )
        assert model.calculate_expected_score(1, -1) == 0.5

    def test_calculate_expected_score_higher_rating(self):
        model = LogisticRegression(
            default_init_rating=1,
            init_ratings=None,
            beta=1,
        )
        assert model.calculate_expected_score(2, -1) > 0.5

    def test_calculate_expected_score_inverse(self):
        model_1 = LogisticRegression(
            default_init_rating=1,
            init_ratings=None,
            beta=1,
        )
        model_2 = LogisticRegression(
            default_init_rating=1,
            init_ratings=None,
            beta=1,
        )
        assert model_1.calculate_expected_score(1, -1) == model_2.calculate_expected_score(-1, 1)


class TestSGDOptimizer:

    def test_calculate_update_step(self):
        model_1 = LogisticRegression(
            default_init_rating=1000,
            init_ratings=dict(entity_1=(None, 1500), entity_2=(None, 1600)),
            beta=200,
        )
        opt_1 = SGDOptimizer(k_factor=32)
        update_1 = opt_1.calculate_update_step(model_1, 1, "entity_1", "entity_2")

        assert round(update_1[0], 2) == 20.48
        assert round(update_1[1], 2) == -20.48

        model_2 = LogisticRegression(
            default_init_rating=1000,
            init_ratings=dict(entity_2=(None, 1600)),
            beta=200,
        )
        opt_2 = SGDOptimizer(k_factor=20)
        update_2 = opt_2.calculate_update_step(model_2, 0, "entity_1", "entity_2")

        assert round(update_2[0], 2) == -0.61
        assert round(update_2[1], 2) == 0.61

    def test_calculate_gradient_raises(self):
        model = LogisticRegression(
            default_init_rating=1000,
            init_ratings=None,
            beta=200,
        )
        opt = SGDOptimizer(k_factor=20)
        with pytest.raises(ValueError, match="Invalid result value"):
            opt.calculate_update_step(model, -1, "entity_1", "entity_2")

    def test_update_model(self):
        model = LogisticRegression(beta=200, default_init_rating=1200, init_ratings=None)
        sgd = SGDOptimizer(k_factor=20)

        sgd.update_model(model, y=1, entity_1="Tom", entity_2="Jerry", t=1)

        assert model.ratings["Tom"] == (1, 1210.0)
        assert model.ratings["Jerry"] == (1, 1190.0)


class TestEloEstimator:

    estimator = EloEstimator(k_factor=20, default_init_rating=1200)

    def test_transform_raises(self):
        with pytest.raises(ValueError, match="X must be a pandas DataFrame."):
            self.estimator.transform(1)

        df = pd.DataFrame(
            columns=["entity_1", "entity_2", "score"],
            index=[3, 2, 1],
        )
        with pytest.raises(ValueError, match="Index must be sorted."):
            self.estimator.transform(df)

    def test_transform(self):
        df = pd.DataFrame(
            data=[
                ("A", "B", 1),
                ("A", "C", 1),
                ("B", "C", 0),
                ("C", "A", 0),
                ("C", "B", 1),
            ],
            columns=["entity_1", "entity_2", "score"],
            index=[1, 2, 3, 4, 4],
        )

        expected_series = pd.Series(
            data=[0.5, 0.51, 0.5, 0.47, 0.53],
            index=[1, 2, 3, 4, 4],
            name="expected_score",
        )

        output_series = self.estimator.transform(df)

        # Check expected scores
        pd.testing.assert_series_equal(expected_series, output_series, check_exact=False, atol=1e-2)

        # Check ratings
        expected_ratings = {
            "A": (2, 1200 + 10 + 9.7123 + 9.4413),
            "B": (1, 1200 - 10 - 9.9917 - 9.4172),
            "C": (2, 1200 - 9.7123 + 9.9917 - 9.4413 + 9.4172),
        }
        for k, v in self.estimator.model.ratings.items():
            assert round(expected_ratings[k][1], 2) == round(v[1], 2)

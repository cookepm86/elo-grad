import pytest

from elo_grad import LogisticRegression, SGDOptimizer


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

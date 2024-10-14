import pytest

from elo_grad import LogisticRegression, SGDOptimizer


class TestLogisticRegression:
    def test_calculate_expected_score_equal_ratings(self):
        model = LogisticRegression(
            default_init_weight=1,
            init_weights=None,
            beta=1,
        )
        assert model.calculate_expected_score(1, -1) == 0.5

    def test_calculate_expected_score_higher_rating(self):
        model = LogisticRegression(
            default_init_weight=1,
            init_weights=None,
            beta=1,
        )
        assert model.calculate_expected_score(2, -1) > 0.5

    def test_calculate_expected_score_inverse(self):
        model_1 = LogisticRegression(
            default_init_weight=1,
            init_weights=None,
            beta=1,
        )
        model_2 = LogisticRegression(
            default_init_weight=1,
            init_weights=None,
            beta=1,
        )
        assert model_1.calculate_expected_score(1, -1) == model_2.calculate_expected_score(-1, 1)


class TestSGDOptimizer:

    def test_calculate_update_step(self):
        opt_1 = SGDOptimizer(
            model=LogisticRegression(
                default_init_weight=1000,
                init_weights=dict(entity_1=(None, 1500), entity_2=(None, 1600)),
                beta=200,
            ),
            alpha=32,
        )
        update_1 = opt_1.calculate_update_step(1, "entity_1", "entity_2")

        assert round(update_1[0], 2) == 20.48
        assert round(update_1[1], 2) == -20.48

        opt_2 = SGDOptimizer(
            model=LogisticRegression(
                default_init_weight=1000,
                init_weights=dict(entity_2=(None, 1600)),
                beta=200,
            ),
            alpha=20,
        )
        update_2 = opt_2.calculate_update_step(0, "entity_1", "entity_2")

        assert round(update_2[0], 2) == -0.61
        assert round(update_2[1], 2) == 0.61

    def test_calculate_gradient_raises(self):
        opt = SGDOptimizer(
            model=LogisticRegression(
                default_init_weight=1000,
                init_weights=None,
                beta=200,
            ),
            alpha=20,
        )
        with pytest.raises(ValueError, match="Invalid result value"):
            opt.calculate_update_step(-1, "entity_1", "entity_2")

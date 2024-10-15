import pytest

from elo_grad import LogisticRegression


class TestLogisticRegression:
    def test_calculate_expected_score_equal_ratings(self):
        model = LogisticRegression(
            alpha=(1600.0, 1600.0),
            beta=200,
        )
        assert model.calculate_expected_result(1, -1) == 0.5

    def test_calculate_expected_score_higher_rating(self):
        model = LogisticRegression(
            alpha=(2000.0, 1600.0),
            beta=200,
        )
        assert model.calculate_expected_result(1, -1) > 0.5

    def test_calculate_expected_score_inverse(self):
        model_1 = LogisticRegression(
            alpha=(2000.0, 1600.0),
            beta=200,
        )
        model_2 = LogisticRegression(
            alpha=(1600.0, 2000.0),
            beta=200,
        )
        assert model_1.calculate_expected_result(1, -1) == model_2.calculate_expected_result(-1, 1)

    def test_calculate_expected_score_raises(self):
        model = LogisticRegression(
            alpha=(2000.0, 1600.0),
            beta=200,
        )
        with pytest.raises(ValueError, match="Length of args/values must match length of alpha/coefficients."):
            model.calculate_expected_result(1)

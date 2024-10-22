# EloGrad

**Extended Elo model implementation.**

**EloGrad** leverages the framing of the 
[Elo rating system](https://en.wikipedia.org/wiki/Elo_rating_system)
as logistic regression with stochastic gradient descent
(see [this blog](https://stmorse.github.io/journal/Elo.html) for a nice walkthrough)
to offer a collection of extensions to the rating system.
All models are `scikit-learn` compatible.

## Installation

You can install `elo-grad` with:
```bash
pip install elo-grad
```

## Quick Start

Detailed example notebooks are provided in the `examples/` directory.
To install any extra dependencies required to run the notebooks install with:
```bash
pip install elo-grad[examples]
```

### Minimal Example

```python
from elo_grad import EloEstimator

# Input DataFrame with sorted index of Unix timestamps
# and columns entity_1 | entity_2 | score
# where score = 1 if player_1 won and score = 0 if
# player_2 won.
df = ...
estimator = EloEstimator(
    k_factor=20, 
    default_init_rating=1200,
    entity_cols=("player_1", "player_2"),
    score_col="result",
)
# Get expected scores
expected_scores = estimator.transform(df)
# Get final ratings (of form (Unix timestamp, rating))
ratings = estimator.model.ratings
```

## Roadmap :compass:

In rough order, things we want to add are:
- Scikit-learn compatibility
- Proper documentation
- Plotting support
- Support for additional features, e.g. home advantage
- Regularization (L1 & L2)
- Support for Polars
- Head-to-head ratings
- Other optimizers, e.g. momentum
- Poisson model support
- Support for draws

## References

1. Elo rating system: https://en.wikipedia.org/wiki/Elo_rating_system
2. Elo rating system as logistic regression with stochastic gradient descent: https://stmorse.github.io/journal/Elo.html

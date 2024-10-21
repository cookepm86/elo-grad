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
uv add git+https://github.com/cookepm86/elo-grad
```

## Quick Start

### Minimal Example

```python
from elo_grad import LogisticRegression, SGDOptimizer

model = LogisticRegression(beta=200, default_init_rating=1200, init_ratings=None)
sgd = SGDOptimizer(k_factor=20)

# Check initial weights (NOTE: time is None)
print("Initial weights:")
print(model.ratings["Tom"], model.ratings["Jerry"])

# Update after Tom beats Jerry at time t=1
sgd.update_model(model, y=1, entity_1="Tom", entity_2="Jerry", t=1)

# Check new weights
print("\nNew weights:")
print(model.ratings["Tom"], model.ratings["Jerry"])
```

Output:
```
Initial weights:
(None, 1200) (None, 1200)

New weights:
(1, 1210.0) (1, 1190.0)
```

## References

1. Elo rating system: https://en.wikipedia.org/wiki/Elo_rating_system
2. Elo rating system as logistic regression with stochastic gradient descent: https://stmorse.github.io/journal/Elo.html

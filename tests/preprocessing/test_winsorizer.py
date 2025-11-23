import numpy as np
import pytest

from ps3.preprocessing import Winsorizer

# TODO: Test your implementation of a simple Winsorizer
@pytest.mark.parametrize(
    "lower_quantile, upper_quantile", [(0, 1), (0.05, 0.95), (0.5, 0.5)]
)
def test_winsorizer(lower_quantile, upper_quantile):

    X = np.random.normal(0, 1, 1000)

    # Reshape to 2D because your transformer expects 2D arrays
    X = X.reshape(-1, 1)

    # Initialize your Winsorizer
    w = Winsorizer(
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile
    )

    # Fit and transform
    w.fit(X)
    X_wins = w.transform(X)

    # Compute expected result using numpy
    lower = np.quantile(X, lower_quantile, axis=0)
    upper = np.quantile(X, upper_quantile, axis=0)
    expected = np.clip(X, lower, upper)

    # The transformed result must match numpy's clipping
    assert np.allclose(X_wins, expected)


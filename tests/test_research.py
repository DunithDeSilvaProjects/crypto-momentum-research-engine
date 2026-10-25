import numpy as np
import pandas as pd

from momentum import research


def test_forward_returns_start_after_the_execution_lag_and_end_in_nan():
    close = pd.DataFrame({"A": [100.0, 200.0, 100.0, 400.0, 100.0]})

    lag1 = research.forward_returns(close, horizon=1, lag=1)["A"].to_numpy()
    lag0 = research.forward_returns(close, horizon=1, lag=0)["A"].to_numpy()

    np.testing.assert_allclose(lag1, [-0.5, 3.0, -0.75, np.nan, np.nan])
    np.testing.assert_allclose(lag0, [1.0, -0.5, 3.0, -0.75, np.nan])
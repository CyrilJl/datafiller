import time

import numpy as np

from datafiller import MultivariateImputer


def test_multivariate_imputer_timing():
    rng = np.random.RandomState(0)
    x = rng.normal(size=(10000, 25)).astype(np.float32)
    missing_mask = rng.rand(*x.shape) < 0.05
    x[missing_mask] = np.nan

    imputer = MultivariateImputer(verbose=0)

    # Warm up JIT compilation.
    _ = imputer(x)

    start = time.perf_counter()
    _ = imputer(x)
    elapsed = time.perf_counter() - start

    print(f"MultivariateImputer elapsed seconds: {elapsed:.6f}")

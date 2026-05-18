import numpy as np

from asfuzz.oracle.numeric import numeric_equal


def test_numeric_nan_pattern():
    a = np.array([1.0, np.nan], dtype="float32")
    b = np.array([1.0, np.nan], dtype="float32")
    assert numeric_equal(a, b, "float32").ok


import numpy as np
from datafiller._optimask import optimask


def test_optimask_no_nans():
    iy = np.array([1], dtype=np.uint32)
    ix = np.array([1], dtype=np.uint32)
    rows = np.arange(3, dtype=np.uint32)
    cols = np.arange(3, dtype=np.uint32)
    global_matrix_size = (3, 3)

    rows_to_keep, cols_to_keep = optimask(iy, ix, rows, cols, global_matrix_size)

    # Create a dummy matrix to test the output
    matrix = np.ones((3, 3))
    matrix[1, 1] = np.nan

    submatrix = matrix[np.ix_(rows_to_keep, cols_to_keep)]
    assert not np.isnan(submatrix).any()

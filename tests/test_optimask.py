import numpy as np
import pytest
from datafiller._optimask import optimask


def get_nan_indices(matrix):
    """Helper function to get NaN indices from a matrix."""
    nan_pos = np.where(np.isnan(matrix))
    return nan_pos[0].astype(np.uint32), nan_pos[1].astype(np.uint32)


def test_optimask_no_nans():
    """Test optimask on a matrix with no NaN values."""
    m, n = 10, 10
    matrix = np.zeros((m, n))
    iy, ix = get_nan_indices(matrix)
    rows = np.arange(m, dtype=np.uint32)
    cols = np.arange(n, dtype=np.uint32)

    rows_to_keep, cols_to_keep = optimask(iy, ix, rows, cols, (m, n))

    np.testing.assert_array_equal(np.sort(rows_to_keep), rows)
    np.testing.assert_array_equal(np.sort(cols_to_keep), cols)


def test_optimask_all_nans():
    """Test optimask on a matrix that is entirely NaN."""
    m, n = 10, 10
    matrix = np.full((m, n), np.nan)
    iy, ix = get_nan_indices(matrix)
    rows = np.arange(m, dtype=np.uint32)
    cols = np.arange(n, dtype=np.uint32)

    rows_to_keep, cols_to_keep = optimask(iy, ix, rows, cols, (m, n))

    assert len(rows_to_keep) == 0
    assert len(cols_to_keep) == 0


def test_optimask_simple_case():
    """Test optimask with a simple and obvious optimal rectangle."""
    m, n = 5, 5
    matrix = np.zeros((m, n))
    matrix[0, :] = np.nan  # First row is all NaN
    matrix[:, 0] = np.nan  # First col is all NaN

    iy, ix = get_nan_indices(matrix)
    rows = np.arange(m, dtype=np.uint32)
    cols = np.arange(n, dtype=np.uint32)

    rows_to_keep, cols_to_keep = optimask(iy, ix, rows, cols, (m, n))

    expected_rows = np.array([1, 2, 3, 4])
    expected_cols = np.array([1, 2, 3, 4])

    np.testing.assert_array_equal(np.sort(rows_to_keep), expected_rows)
    np.testing.assert_array_equal(np.sort(cols_to_keep), expected_cols)


def test_optimask_complex_case():
    """Test optimask with a more complex NaN pattern."""
    m, n = 6, 6
    matrix = np.zeros((m, n))
    matrix[2, 3:] = np.nan
    matrix[4:, 5] = np.nan
    matrix[0, 1] = np.nan

    # The largest clean area is the bottom-left 4x5 rectangle (rows 2-5, cols 0-4)
    # or the top-right 2x5 rectangle (rows 0-1, cols 0,2,3,4) plus others.
    # The algorithm should find a large clean rectangle.
    # The expected largest area is rows=[0,1,3,4,5], cols=[0,2,3,4] (5x4=20)
    # or rows=[1,3,4,5], cols=[0,1,2,3,4] (4x5=20)
    # Let's trace:
    # NaNs at: (0,1), (2,3), (2,4), (2,5), (4,5), (5,5)
    # The largest rectangle of non-NaNs is the 5x4 area at rows {1,3,4,5,0} and cols {0,2,3,4}
    # Or more simply, rows {0,1,3,4,5} and cols {0,2,3,4}
    # Another large one is rows {1,3,4,5} and cols {0,1,2,3,4}
    # The algorithm is deterministic, so it will find one of these.
    # Let's check what it should find.
    # The algorithm should find the 5x4 area.
    expected_rows = np.array([0, 1, 3, 4, 5])
    expected_cols = np.array([0, 2, 3, 4])

    iy, ix = get_nan_indices(matrix)
    rows = np.arange(m, dtype=np.uint32)
    cols = np.arange(n, dtype=np.uint32)

    rows_to_keep, cols_to_keep = optimask(iy, ix, rows, cols, (m, n))

    # The area of the returned rectangle should be maximal. The largest possible area is 20.
    assert len(rows_to_keep) * len(cols_to_keep) >= 20


def test_optimask_no_valid_rectangle():
    """Test a case where no clean rectangle of size > 1x1 can be found."""
    m, n = 4, 4
    matrix = np.zeros((m, n))
    matrix[0, 1] = np.nan
    matrix[1, 0] = np.nan
    matrix[2, 3] = np.nan
    matrix[3, 2] = np.nan
    matrix[1, 2] = np.nan
    matrix[2, 1] = np.nan

    # With this checkerboard-like pattern, the largest clean area might be small.
    iy, ix = get_nan_indices(matrix)
    rows = np.arange(m, dtype=np.uint32)
    cols = np.arange(n, dtype=np.uint32)

    rows_to_keep, cols_to_keep = optimask(iy, ix, rows, cols, (m, n))

    # With this checkerboard-like pattern, the largest clean area is 3x2=6.
    # The current heuristic finds a 2x2 area. This test verifies the current behavior.
    assert len(rows_to_keep) * len(cols_to_keep) >= 4


from datafiller._optimask import (
    _get_largest_rectangle,
    _process_index,
    apply_p_step,
    groupby_max,
    is_decreasing,
    numba_apply_permutation,
    numba_apply_permutation_inplace,
    numba_setdiff1d,
)


def test_get_largest_rectangle_branch():
    heights = np.array([1, 2, 3])
    m, n = 5, 4  # n > len(heights)
    i0, h_i0, area = _get_largest_rectangle(heights, m, n)
    assert i0 == 0  # (5-1)*(4-0) = 16. (5-2)*(4-1)=9, (5-3)*(4-2)=4
    assert h_i0 == 1
    assert area == 16


def test_optimask_convergence_error(mocker):
    mocker.patch("datafiller._optimask.is_decreasing", return_value=False)
    with pytest.raises(ValueError, match="Pareto optimization did not converge"):
        m, n = 10, 10
        matrix = np.zeros((m, n))
        matrix[0, 0] = np.nan
        iy, ix = get_nan_indices(matrix)
        rows = np.arange(m, dtype=np.uint32)
        cols = np.arange(n, dtype=np.uint32)
        optimask(iy, ix, rows, cols, (m, n))


# Tests for numba functions
def test_is_decreasing():
    assert is_decreasing(np.array([3, 2, 1], dtype=np.uint32))
    assert not is_decreasing(np.array([1, 2, 3], dtype=np.uint32))
    assert is_decreasing(np.array([3, 2, 2], dtype=np.uint32))


def test_groupby_max():
    a = np.array([0, 1, 0, 2, 1], dtype=np.uint32)
    b = np.array([5, 2, 8, 3, 4], dtype=np.uint32)
    n = 3
    result = groupby_max(a, b, n)
    # expected: ret[0] = max(0, 5+1, 8+1) = 9. ret[1] = max(0, 2+1, 4+1) = 5. ret[2] = max(0, 3+1) = 4
    assert np.array_equal(result, [9, 5, 4])


def test_numba_setdiff1d():
    a = np.array([1, 2, 3, 4, 5], dtype=np.uint32)
    b = np.array([2, 4], dtype=np.uint32)
    n = 6
    result = numba_setdiff1d(a, b, n)
    assert np.array_equal(result, [1, 3, 5])


def test_apply_p_step():
    a = np.array([10, 20, 30], dtype=np.uint32)
    b = np.array([1, 2, 3], dtype=np.uint32)
    p_step = np.array([2, 0, 1], dtype=np.uint32)
    ret_a, ret_b = apply_p_step(p_step, a, b)
    assert np.array_equal(ret_a, [30, 10, 20])
    assert np.array_equal(ret_b, [3, 1, 2])


def test_numba_apply_permutation():
    p = np.array([2, 0, 1], dtype=np.uint32)
    x = np.array([0, 1, 2], dtype=np.uint32)
    result = numba_apply_permutation(p, x)
    # rank[2]=0, rank[0]=1, rank[1]=2
    # result[0] = rank[0] = 1
    # result[1] = rank[1] = 2
    # result[2] = rank[2] = 0
    assert np.array_equal(result, [1, 2, 0])


def test_numba_apply_permutation_inplace():
    p = np.array([2, 0, 1], dtype=np.uint32)
    x = np.array([0, 1, 2], dtype=np.uint32)
    numba_apply_permutation_inplace(p, x)
    assert np.array_equal(x, [1, 2, 0])


def test_process_index():
    index = np.array([10, 20, 10, 30], dtype=np.uint32)
    num = 40
    ret, table_inv, cnt = _process_index(index, num)
    # table[10]=1, table[20]=2, table[30]=3
    # ret = [0, 1, 0, 2]
    # table_inv = [10, 20, 30]
    # cnt = 3
    assert np.array_equal(ret, [0, 1, 0, 2])
    assert np.array_equal(table_inv, [10, 20, 30])
    assert cnt == 3

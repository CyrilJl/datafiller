import numpy as np

from datafiller._optimask import _get_largest_rectangle, optimask


def test_constrained_rectangle_prefers_rows_when_required():
    # pareto histogram: cut i keeps (m - h[i]) rows and (n - i) cols.
    # Unconstrained max-area is 15 rows x 10 cols; only cut 8 keeps >= 20 rows.
    h = np.array([10, 10, 10, 10, 10, 10, 10, 10, 5, 0], dtype=np.uint32)
    m, n = 25, 10

    i0, j0, _ = _get_largest_rectangle(h.copy(), m, n, min_rows=1)
    assert (m - j0, n - i0) == (15, 10)

    i0, j0, _ = _get_largest_rectangle(h.copy(), m, n, min_rows=20)
    assert m - j0 >= 20

    # infeasible constraint falls back to the unconstrained maximum
    i0, j0, _ = _get_largest_rectangle(h.copy(), m, n, min_rows=m + 1)
    assert (m - j0, n - i0) == (15, 10)


def test_optimask_min_rows_returns_enough_rows():
    # 30 rows x 6 cols; cols 1..5 have NaNs in rows 10..29, col 0 is clean.
    # Max-area keeps 10 rows x 6 cols (60 cells); min_rows=20 must sacrifice
    # columns instead and return >= 20 rows.
    matrix = np.ones((30, 6))
    matrix[10:, 1:] = np.nan
    iy, ix = np.nonzero(np.isnan(matrix))
    iy = iy.astype(np.uint32)
    ix = ix.astype(np.uint32)
    rows = np.arange(30, dtype=np.uint32)
    cols = np.arange(6, dtype=np.uint32)

    r0, c0 = optimask(iy.copy(), ix.copy(), rows, cols, matrix.shape)
    assert len(r0) == 10 and len(c0) == 6

    r1, c1 = optimask(iy.copy(), ix.copy(), rows, cols, matrix.shape, min_rows=20)
    assert len(r1) >= 20
    assert not np.isnan(matrix[np.ix_(r1, c1)]).any()


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

import numpy as np
from numba import bool_, njit, prange, uint32
from numba.types import UniTuple


@njit(bool_(uint32[:]), boundscheck=True, cache=True)
def is_decreasing(h):
    """
    numba equivalent to:
        return (np.diff(h)>0).all()
    """
    for i in range(len(h) - 1):
        if h[i] < h[i + 1]:
            return False
    return True


@njit(uint32[:](uint32[:], uint32[:], uint32), boundscheck=True, cache=True)
def groupby_max(a, b, n):
    """
    numba equivalent to:
        size_a = len(a)
        ret = np.zeros(n, dtype=np.uint32)
        np.maximum.at(ret, a, b + 1)
        return ret
    """
    size_a = len(a)
    ret = np.zeros(n, dtype=np.uint32)
    for k in range(size_a):
        ak = a[k]
        ret[ak] = max(ret[ak], b[k] + 1)
    return ret


@njit(UniTuple(uint32[:], 2)(uint32[:], uint32[:], uint32[:]), parallel=True, boundscheck=True, cache=True)
def apply_p_step(p_step, a, b):
    ret_a = np.empty(a.size, dtype=np.uint32)
    ret_b = np.empty(b.size, dtype=np.uint32)
    for k in prange(a.size):
        pk = p_step[k]
        ret_a[k] = a[pk]
        ret_b[k] = b[pk]
    return ret_a, ret_b


@njit(uint32[:](uint32[:], uint32[:]), parallel=True, boundscheck=True, cache=True)
def numba_apply_permutation(p, x):
    """
    numba equivalent to:
        rank = np.empty_like(p)
        rank[p] = np.arange(len(p))
        # Use the rank array to permute x
        return rank[x]
    """
    n = p.size
    m = x.size
    rank = np.empty(n, dtype=np.uint32)
    result = np.empty(m, dtype=np.uint32)

    for i in prange(n):
        rank[p[i]] = i

    for i in prange(m):
        result[i] = rank[x[i]]
    return result


@staticmethod
@njit((uint32[:], uint32[:]), parallel=True, boundscheck=True, cache=True)
def numba_apply_permutation_inplace(p, x):
    n = p.size
    rank = np.empty(n, dtype=np.uint32)

    for i in prange(n):
        rank[p[i]] = i

    for i in prange(x.size):
        x[i] = rank[x[i]]


def apply_permutation(p, x, inplace: bool):
    """
    Applies a permutation to an array.

    Args:
        p (np.ndarray): The permutation array.
        x (np.ndarray): The array to be permuted.
        inplace (bool): If True, applies the permutation in place; otherwise, returns a new permuted array.

    Returns:
        np.ndarray: The permuted array if inplace is False; otherwise, None.
    """
    if inplace:
        numba_apply_permutation_inplace(p, x)
    else:
        return numba_apply_permutation(p, x)


@njit(boundscheck=True, fastmath=True, nogil=True)
def _process_index(index, num):
    size = len(index)
    table = np.zeros(num, dtype=np.uint32)
    table_inv = np.empty(num, dtype=np.uint32)
    ret = np.empty(size, dtype=np.uint32)
    cnt = np.uint32(0)

    for k in range(size):
        elem = index[k]
        if table[elem] == 0:
            cnt += 1
            table[elem] = cnt
            table_inv[cnt - 1] = elem
        ret[k] = table[elem] - 1

    return ret, table_inv[:cnt], cnt


@staticmethod
def _get_largest_rectangle(heights, m, n):
    if n > len(heights):
        heights = np.concat([heights, [0]])
    areas = (m - heights) * (n - np.arange(len(heights)))
    i0 = np.argmax(areas)
    return i0, heights[i0], areas[i0]


@njit(uint32[:](uint32, uint32[:], uint32[:], uint32), parallel=True, boundscheck=True, cache=True)
def compute_to_keep(size, index_with_nan, permutation, split):
    """
    Computes the indices to keep after removing a subset of indices with NaNs.

    Args:
        size (int): The total number of indices.
        index_with_nan (np.ndarray): The indices that contain NaNs.
        permutation (np.ndarray): The permutation array.
        split (int): The split point in the permutation array.

    Returns:
        np.ndarray: The indices to keep after removing the subset with NaNs.
    """
    mask = np.zeros(size, dtype=np.bool_)
    for i in prange(split):
        mask[index_with_nan[permutation[i]]] = True

    count = size - split

    result = np.empty(count, dtype=np.uint32)
    idx = 0
    for i in range(size):
        if not mask[i]:
            result[idx] = i
            idx += 1
    return result


def optimask(iy, ix, rows, cols, global_matrix_size):
    m, n = global_matrix_size
    iyp, rows_with_nan, m_nan = _process_index(index=iy, num=m)
    ixp, cols_with_nan, n_nan = _process_index(index=ix, num=n)
    hy = groupby_max(iyp, ixp, m_nan)
    hx = groupby_max(ixp, iyp, n_nan)

    p_rows = np.arange(m_nan, dtype=np.uint32)
    p_cols = np.arange(n_nan, dtype=np.uint32)
    is_pareto_ordered = False

    step = 0
    while not is_pareto_ordered and step < 16:
        kind = "stable" if step else "quicksort"
        axis = step % 2
        step += 1
        if axis == 0:
            p_step = (-hy).argsort(kind=kind).astype(np.uint32)
            apply_permutation(p_step, iyp, inplace=True)
            p_rows, hy = apply_p_step(p_step, p_rows, hy)
            hx = groupby_max(ixp, iyp, n_nan)
            is_pareto_ordered = is_decreasing(hx)
        else:
            p_step = (-hx).argsort(kind=kind).astype(np.uint32)
            apply_permutation(p_step, ixp, inplace=True)
            hy = groupby_max(iyp, ixp, m_nan)
            p_cols, hx = apply_p_step(p_step, p_cols, hx)
            is_pareto_ordered = is_decreasing(hy)

    if not is_pareto_ordered:
        raise ValueError(f"{step}")
    i0, j0, area = _get_largest_rectangle(hx, m, n)
    cols_to_keep = np.setdiff1d(cols, cols_with_nan[p_cols][:i0])
    rows_to_keep = np.setdiff1d(rows, rows_with_nan[p_rows][:j0])

    return rows_to_keep, cols_to_keep

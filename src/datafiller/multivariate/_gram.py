"""Per-missingness-pattern ridge solves from cached Gram matrices.

The imputer trains one model per observed-feature pattern of the rows it has to
predict, on the training rows that are complete over that pattern's features.
Those training sets overlap heavily: a row whose NaNs sit in columns `N` is a
valid training row for *every* pattern that excludes a superset of `N`. Rebuilding
each pattern's Gram matrix from its own row subset therefore re-accumulates the
same rows over and over — on a representative benchmark, 3.1 GFLOP of outer
products where 0.11 GFLOP of distinct information exists.

This module groups the training rows by their exact NaN pattern, accumulates one
Gram matrix per group once, and then assembles each missingness pattern's Gram as
`gram_complete + sum of the groups it admits`. Groups with a single row get no
cached matrix (there is nothing to reuse) and neither do groups beyond a memory
budget; those rows are accumulated on demand, so wide inputs -- where nearly every
row has a distinct NaN pattern -- degrade to the direct method instead of trying to
cache tens of thousands of matrices.

Both lookups are driven off each row's or group's *lowest* NaN column: a pattern
enumerates candidates from its excluded columns through the NaN index, so its cost
follows the number of NaNs in those columns rather than the number of training rows.
"""

import numpy as np
from numba import njit, prange

#: Memory ceiling for the cached per-group Gram matrices. Groups beyond what fits
#: are handled by direct row accumulation.
_GRAM_CACHE_BUDGET_BYTES = 64_000_000

# Gram matrices are accumulated and solved in float64 throughout. A pattern's Gram
# is a sum of many group contributions, and the intercept correction
# (`sxx - outer(sx, sx) / n`) cancels most of their magnitude, so single precision
# here loses several digits on data that is not centered.


@njit(boundscheck=False, cache=True)
def pack_bit_words(mask: np.ndarray, n_bits: int) -> np.ndarray:
    """Pack the first `n_bits` columns of each row of a boolean matrix into 64-bit words.

    Turns per-row column sets into bitmasks so that "are all of this row's NaNs
    inside this pattern's excluded columns" becomes a couple of integer
    operations instead of a scan over the columns. Taking `n_bits` explicitly
    lets callers pass a wider contiguous mask instead of slicing a strided view
    out of it.
    """
    m = mask.shape[0]
    n_words = (n_bits + 63) // 64
    words = np.zeros((m, n_words), dtype=np.uint64)
    one = np.uint64(1)
    for i in range(m):
        row = mask[i]
        for j in range(n_bits):
            if row[j]:
                words[i, j >> 6] |= one << np.uint64(j & 63)
    return words


@njit(boundscheck=False, cache=True, parallel=True)
def gather_augmented(x: np.ndarray, rows: np.ndarray, cols: np.ndarray, target_col: int) -> np.ndarray:
    """Gather the augmented training matrix `[X, y, 1]` in a single pass.

    Building it directly from `x` avoids materializing the feature subset and
    the target column as separate arrays first; the caller reads them back as
    views into the result.
    """
    n_rows = len(rows)
    n_cols = len(cols)
    out = np.empty((n_rows, n_cols + 2), dtype=np.float32)
    for i in prange(n_rows):  # ty: ignore[not-iterable]
        source = x[rows[i]]
        target = out[i]
        for j in range(n_cols):
            target[j] = np.float32(source[cols[j]])
        target[n_cols] = np.float32(source[target_col])
        target[n_cols + 1] = np.float32(1.0)
    return out


@njit(boundscheck=False, cache=True, parallel=True)
def group_grams(z_aug: np.ndarray, group_rows: np.ndarray, group_ptr: np.ndarray, width: int) -> np.ndarray:
    """Gram matrix of `z_aug` restricted to each group of rows.

    NaN cells contribute zero. That is exact for the way these Grams are used:
    a group only ever contributes to patterns that exclude every column where
    the group has NaNs, so those rows and columns are dropped from the solve.
    """
    n_groups = len(group_ptr) - 1
    grams = np.zeros((n_groups, width, width), dtype=np.float64)
    for g in prange(n_groups):  # ty: ignore[not-iterable]
        gram = grams[g]
        values = np.empty(width, dtype=np.float64)
        for t in range(group_ptr[g], group_ptr[g + 1]):
            row = z_aug[group_rows[t]]
            for a in range(width):
                v = row[a]
                values[a] = 0.0 if np.isnan(v) else np.float64(v)
            for a in range(width):
                va = values[a]
                if va != 0.0:
                    gram_a = gram[a]
                    for b in range(width):
                        gram_a[b] += va * values[b]
    return grams


def complete_gram(z_aug: np.ndarray, rows: np.ndarray, width: int, chunk: int = 8192) -> np.ndarray:
    """Double-precision Gram matrix of the fully observed training rows.

    Accumulated in row chunks so the double-precision working copy stays cache
    sized instead of doubling the augmented matrix.
    """
    gram = np.zeros((width, width), dtype=np.float64)
    full = len(rows) == len(z_aug)
    for start in range(0, len(rows), chunk):
        block = (z_aug[start : start + chunk] if full else z_aug[rows[start : start + chunk]]).astype(np.float64)
        gram += block.T @ block
    return gram


class RowGroups:
    """Training rows grouped by their exact NaN pattern.

    Attributes:
        words: Bitmask of each cached group's NaN columns, shape `(n_groups, n_words)`.
        counts: Number of rows in each cached group.
        rows: Cached group members, concatenated in group order.
        offsets: CSR offsets into `rows`.
        by_column_offsets: CSR offsets indexing groups by their lowest NaN column.
        by_column_ids: Group ids ordered by their lowest NaN column.
        row_words: Bitmask of every row's NaN columns.
        row_group: Cached group of each row, or -1 when the row is not cached.
    """

    __slots__ = (
        "words",
        "counts",
        "rows",
        "offsets",
        "by_column_offsets",
        "by_column_ids",
        "row_words",
        "row_group",
    )

    def __init__(self, words, counts, rows, offsets, by_column_offsets, by_column_ids, row_words, row_group):
        self.words = words
        self.counts = counts
        self.rows = rows
        self.offsets = offsets
        self.by_column_offsets = by_column_offsets
        self.by_column_ids = by_column_ids
        self.row_words = row_words
        self.row_group = row_group


def build_row_groups(
    mask_nan_local: np.ndarray,
    n_features: int,
    row_has_nan: np.ndarray,
    first_nan_column: np.ndarray,
    width: int,
) -> RowGroups:
    """Group the training rows that hold NaNs by their exact NaN pattern.

    Only groups with at least two rows get a cached Gram matrix: a single-row
    group offers no reuse, so accumulating it on demand is cheaper than storing
    a full matrix for it. The cache is additionally capped by a memory budget.
    Groups are indexed by their lowest NaN column, which lets a pattern find its
    candidates from its excluded columns instead of scanning every group.

    Args:
        mask_nan_local: NaN mask of the local training matrix; only its first
            `n_features` columns are considered.
        n_features: Number of feature columns.
        row_has_nan: Whether each row holds at least one NaN.
        first_nan_column: Lowest NaN column of each row (any value for rows
            without NaNs).
        width: Side length of the augmented Gram matrices, used to size the cache.
    """
    words = pack_bit_words(mask_nan_local, n_features)
    n_rows = len(words)
    rows_with_nan = np.flatnonzero(row_has_nan).astype(np.uint32, copy=False)
    row_group = np.full(n_rows, -1, dtype=np.int64)
    empty_by_column = np.zeros(n_features + 1, dtype=np.int64)
    if not rows_with_nan.size:
        return RowGroups(
            np.zeros((0, words.shape[1]), dtype=np.uint64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.uint32),
            np.zeros(1, dtype=np.int64),
            empty_by_column,
            np.zeros(0, dtype=np.int64),
            words,
            row_group,
        )

    candidate_words = words[rows_with_nan]
    if candidate_words.shape[1] == 1:
        # `np.unique(..., axis=0)` falls back to a lexicographic sort of a void
        # view; up to 64 features the bitmask is a single integer, which sorts
        # an order of magnitude faster.
        flat, inverse, counts = np.unique(candidate_words[:, 0], return_inverse=True, return_counts=True)
        unique_words = flat.reshape(-1, 1)
    else:
        unique_words, inverse, counts = np.unique(candidate_words, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()

    max_groups = max(1, _GRAM_CACHE_BUDGET_BYTES // (width * width * 8))
    kept = np.flatnonzero(counts >= 2)
    if len(kept) > max_groups:
        kept = np.sort(kept[np.argsort(-counts[kept], kind="stable")[:max_groups]])

    remap = np.full(len(unique_words), -1, dtype=np.int64)
    remap[kept] = np.arange(len(kept))
    group_ids = remap[inverse]
    cached = group_ids >= 0
    cached_rows = rows_with_nan[cached]
    cached_ids = group_ids[cached]
    order = np.argsort(cached_ids, kind="stable")
    group_rows = cached_rows[order]
    group_counts = np.bincount(cached_ids, minlength=len(kept)).astype(np.int64)
    offsets = np.zeros(len(kept) + 1, dtype=np.int64)
    np.cumsum(group_counts, out=offsets[1:])
    row_group[cached_rows] = cached_ids

    group_first_column = (
        first_nan_column[group_rows[offsets[:-1]]].astype(np.int64) if len(kept) else np.zeros(0, dtype=np.int64)
    )
    by_column_offsets = np.zeros(n_features + 1, dtype=np.int64)
    np.cumsum(np.bincount(group_first_column, minlength=n_features), out=by_column_offsets[1:])
    return RowGroups(
        np.ascontiguousarray(unique_words[kept]),
        group_counts,
        group_rows,
        offsets,
        by_column_offsets,
        np.argsort(group_first_column, kind="stable").astype(np.int64),
        words,
        row_group,
    )


@njit(boundscheck=False, cache=True, parallel=True)
def solve_patterns(
    z_aug: np.ndarray,
    gram_complete: np.ndarray,
    n_complete: int,
    group_grams_: np.ndarray,
    group_words: np.ndarray,
    group_counts: np.ndarray,
    group_by_column_offsets: np.ndarray,
    group_by_column_ids: np.ndarray,
    row_words: np.ndarray,
    row_group: np.ndarray,
    first_nan_column: np.ndarray,
    nan_col_offsets: np.ndarray,
    nan_col_rows: np.ndarray,
    patterns: np.ndarray,
    excluded_words: np.ndarray,
    local_predict: np.ndarray,
    predict_flat: np.ndarray,
    predict_ptr: np.ndarray,
    imputable_rows: np.ndarray,
    x_imputed: np.ndarray,
    col: int,
    alpha: float,
    fit_intercept: bool,
    min_samples_train: int,
    norm_mean: float,
    norm_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve and apply the ridge model of every missingness pattern.

    For each pattern, assembles the Gram matrix of the training rows complete on
    its features, solves the ridge system on the pattern's usable columns, and
    writes the predictions into `x_imputed` (rescaled to the original units).

    A training row qualifies for a pattern exactly when all of its NaNs fall in
    the pattern's excluded columns. Candidates are therefore enumerated from the
    excluded columns through the NaN index, and each is attributed to its lowest
    NaN column so it is visited once; the cost per pattern follows the number of
    NaNs in the excluded columns rather than the number of training rows.

    Returns:
        `(n_samples, solved)`: how many training rows each pattern found, and
        whether it reached `min_samples_train` and was therefore imputed here.
        Unsolved patterns are left to the caller's `optimask` fallback.
    """
    n_patterns, k = patterns.shape
    width = k + 2
    n_words = group_words.shape[1]
    n_groups = len(group_counts)
    n_rows = len(row_words)

    n_samples = np.zeros(n_patterns, dtype=np.int64)
    solved = np.zeros(n_patterns, dtype=np.bool_)

    # Patterns are independent: each owns its scratch buffers and writes to a
    # disjoint set of prediction rows.
    for p in prange(n_patterns):  # ty: ignore[not-iterable]
        gram = np.empty((width, width), dtype=np.float64)
        usable = np.empty(k, dtype=np.int64)
        matched_groups = np.empty(max(n_groups, 1), dtype=np.int64)
        matched_rows = np.empty(max(n_rows, 1), dtype=np.int64)
        n_usable = 0
        for j in range(k):
            if patterns[p, j]:
                usable[n_usable] = j
                n_usable += 1
        if n_usable == 0:
            continue

        allowed = excluded_words[p]
        total = n_complete
        n_matched = 0
        n_matched_rows = 0
        for j in range(k):
            if patterns[p, j]:
                continue  # not excluded, so nothing here can qualify
            for idx in range(group_by_column_offsets[j], group_by_column_offsets[j + 1]):
                g = group_by_column_ids[idx]
                group = group_words[g]
                ok = True
                for w in range(n_words):
                    if group[w] & ~allowed[w] != np.uint64(0):
                        ok = False
                        break
                if ok:
                    matched_groups[n_matched] = g
                    n_matched += 1
                    total += group_counts[g]
            for t in range(nan_col_offsets[j], nan_col_offsets[j + 1]):
                r = nan_col_rows[t]
                if row_group[r] >= 0 or first_nan_column[r] != j:
                    continue  # counted with its cached group, or reached via another column
                candidate = row_words[r]
                ok = True
                for w in range(n_words):
                    if candidate[w] & ~allowed[w] != np.uint64(0):
                        ok = False
                        break
                if ok:
                    matched_rows[n_matched_rows] = r
                    n_matched_rows += 1
        total += n_matched_rows

        n_samples[p] = total
        if total < min_samples_train:
            continue

        for a in range(width):
            source = gram_complete[a]
            target = gram[a]
            for b in range(width):
                target[b] = source[b]
        for t in range(n_matched):
            contribution = group_grams_[matched_groups[t]]
            for a in range(width):
                source = contribution[a]
                target = gram[a]
                for b in range(width):
                    target[b] += source[b]
        for t in range(n_matched_rows):
            row = z_aug[matched_rows[t]]
            for a in range(width):
                va = row[a]
                if not np.isnan(va) and va != 0.0:
                    target = gram[a]
                    for b in range(width):
                        vb = row[b]
                        if not np.isnan(vb):
                            target[b] += np.float64(va) * np.float64(vb)

        matrix = np.empty((n_usable, n_usable), dtype=np.float64)
        rhs = np.empty(n_usable, dtype=np.float64)
        sum_x = np.empty(n_usable, dtype=np.float64)
        n_train = np.float64(total)
        sum_y = gram[k, k + 1]
        for a in range(n_usable):
            source = gram[usable[a]]
            target = matrix[a]
            for b in range(n_usable):
                target[b] = source[usable[b]]
            rhs[a] = source[k]
            sum_x[a] = source[k + 1]
        if fit_intercept:
            mean_y = sum_y / n_train
            for a in range(n_usable):
                target = matrix[a]
                for b in range(n_usable):
                    target[b] -= sum_x[a] * sum_x[b] / n_train
                rhs[a] -= sum_x[a] * mean_y
        for a in range(n_usable):
            matrix[a, a] += alpha
        coef = np.linalg.solve(matrix, rhs)
        if fit_intercept:
            adjustment = 0.0
            for a in range(n_usable):
                adjustment += (sum_x[a] / n_train) * coef[a]
            intercept = sum_y / n_train - adjustment
        else:
            intercept = 0.0

        for t in range(predict_ptr[p], predict_ptr[p + 1]):
            r = predict_flat[t]
            row = local_predict[r]
            accumulator = 0.0
            for a in range(n_usable):
                accumulator += np.float64(row[usable[a]]) * coef[a]
            x_imputed[imputable_rows[r], col] = (accumulator + intercept) * norm_scale + norm_mean
        solved[p] = True

    return n_samples, solved

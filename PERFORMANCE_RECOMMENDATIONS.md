# Performance Recommendations

Benchmarks were run from `.cache/perf_bottlenecks/bench_datafiller_bottlenecks.py` with explicit Numba warmup excluded from timed measurements. Time is the primary optimization target; memory reductions are secondary unless they also reduce runtime.

## Summary

The main bottleneck is not an individual Numba kernel. It is the number of times the imputer calls `optimask` and then fits a model for each missingness pattern.

In the measured end-to-end cases:

- `mv_6k36_all_features_cols12`: median `0.519s`; `optimask` accounts for about `69%` of runtime.
- `mv_8k96_nearest24_cols16`: median `0.386s`; `optimask` accounts for about `61%` of runtime.
- `FastRidge.fit` is the second bottleneck, around `22-25%` of runtime.
- `_subset` is measurable but lower priority, around `7-10%` of runtime.

## Priority 1: Reduce `optimask` Calls

### Add a Complete-Case Fast Path (DONE)

Current flow in `MultivariateImputer._impute_col`:

```python
mask_usable_cols = _index_to_mask(usable_cols_local, len(sampled_cols_uint32))
iy_trial, ix_trial = nan_positions_subset_cols(local_iy, local_ix, mask_usable_cols)
rows, cols = optimask(...)
```

Recommended flow:

```python
rows = complete_rows_for_cols(local_nan_mask, usable_cols_local)

if len(rows) >= self.min_samples_train:
    cols = usable_cols_local
else:
    mask_usable_cols = _index_to_mask(usable_cols_local, len(sampled_cols_uint32))
    iy_trial, ix_trial = nan_positions_subset_cols(local_iy, local_ix, mask_usable_cols)
    rows, cols = optimask(...)
```

Rationale:

- For low and moderate missingness, many patterns have enough complete training rows.
- This avoids both `nan_positions_subset_cols` and `optimask`.
- It preserves current behavior as a fallback when complete-case rows are insufficient.

Suggested implementation:

- Add a Numba helper like `complete_rows_for_cols(mask_nan, cols)`.
- Build `local_mask_nan = np.isnan(local_train)` once per target column.
- Use the complete-case fast path before constructing `iy_trial` / `ix_trial`.

Expected impact:

- High. This directly targets the measured `61-69%` runtime bottleneck.

## Priority 2: Reduce Pattern and Model Count

### Cap or Coarsen Prediction Patterns

The current implementation treats each unique prediction missingness pattern independently:

```python
patterns, indexes = unique2d(~np.isnan(local_predict))
prediction_groups = self._group_pattern_rows(indexes)
```

Each pattern can cause:

- one `optimask` call,
- one training subset extraction,
- one `FastRidge.fit`,
- one prediction subset extraction.

Recommended approach:

- Add an optional parameter such as `max_patterns_per_column`.
- Keep the most common `K` usable-column patterns.
- Map rare patterns to a retained pattern that is a subset of available columns.
- Fall back to exact-pattern behavior if no compatible retained pattern exists.

Rationale:

- This reduces both `optimask` calls and estimator fits.
- It is a larger behavior change than the complete-case fast path, so it should be configurable and benchmarked for quality impact.

Expected impact:

- Potentially high, especially for wide matrices or noisy missingness where many rare patterns occur.

## Priority 3: JIT the Full `optimask` Wrapper

Most helpers inside `_optimask.py` are Numba-jitted, but the public `optimask` function is regular Python and dispatches many small jitted helpers.

Recommended approach:

- Move the body of `optimask` into a fully jitted internal function, e.g. `_optimask_numba`.
- Keep `optimask` as a thin public wrapper.

Rationale:

- `optimask` is called thousands of times per imputation in the benchmarked cases.
- Even small Python dispatch overhead accumulates.

Expected impact:

- Medium. This is useful, but less important than avoiding the calls entirely.

## Priority 4: Reduce Scratch Allocations in `optimask` (NOT ENOUGH GAINS OBSERVED)

After reducing call count, optimize repeated allocations inside `_optimask.py`.

Targets:

- `_process_index` allocates tables and compressed index arrays per call.
- `(-hy).argsort(...)` allocates a temporary negated array.
- `diff1d` allocates `to_exclude = np.zeros(max_val)` per call.
- `apply_p_step` allocates new arrays each iteration.

Recommended approach:

- Reuse scratch buffers inside one `_impute_col` where possible.
- Avoid negated sort temporaries if a descending sort helper is introduced.
- Replace `diff1d` scratch masks with reusable marker arrays.

Expected impact:

- Medium to low for time after call-count reductions.
- Useful for memory and allocator pressure.

## Lower Priority: `_subset`

`_subset` is already faster than NumPy advanced indexing in the isolated benchmarks:

- `_subset_train_18k40`: Numba `0.59ms`, NumPy `1.79ms`.
- `_subset_predict_2k20`: Numba `0.038ms`, NumPy `0.111ms`.

It still accounts for `7-10%` end-to-end because it is called frequently. Optimize it only after reducing pattern count and `optimask` calls.

Potential follow-up:

- Avoid materializing full `local_train` if the final train subsets can be gathered directly from `x`.
- Reuse allocated arrays for repeated shapes.

## Memory Recommendations

Memory is secondary, but several allocations are unnecessary.

### `nan_positions`

Current implementation allocates `iy` and `ix` with length `m * n`, then slices to the actual NaN count.

Measured examples:

- Sparse `25k x 80`, 2% NaNs: Numba peak `17.17 MB`; NumPy alternative peak `4.43 MB`; returned payload `2.21 MB`.
- Dense `12k x 120`, 15% NaNs: Numba peak `12.36 MB`; NumPy alternative peak `6.31 MB`; returned payload `3.02 MB`.

Recommendation:

- Use a two-pass Numba implementation: first count NaNs, then allocate exact-size `iy` and `ix`.
- This may be slightly slower, so keep it behind a benchmark decision if time remains the strict priority.

### `scoring`

`scoring` creates multiple full-matrix temporaries:

- `isfinite`
- `xp`
- `nan_mask`
- `xp_standard`

Measured on `20k x 160`:

- input/result matrix size: `12.21 MB`
- `scoring` peak: `51.17 MB`
- returned score matrix: `0.018 MB`

Recommendation:

- Standardize in place on a copied preimputed matrix.
- Avoid retaining both preimputed and standardized full-size matrices.
- Consider computing correlations in chunks for very wide inputs.

### Time-Series Lag Expansion

Measured `TimeSeriesImputer` case:

- end-to-end peak: `130.06 MB`
- final output memory: `2.35 MB`
- lag frame construction: `21.81 MB`
- `.values` extraction after concat: `16.03 MB`
- multivariate call on lagged values: `116.26 MB`

Recommendation:

- Keep this lower priority for time, because lag construction itself was only about `1.5ms`.
- For memory-sensitive workloads, avoid retaining both `DataFrame` lag columns and dense NumPy copies longer than necessary.

## Recommended Implementation Order

1. Add `complete_rows_for_cols` and use it as a fast path before `optimask`.
2. Benchmark end-to-end with the existing `.cache/perf_bottlenecks` script.
3. If more speed is needed, add optional pattern coarsening with `max_patterns_per_column`.
4. JIT the full `optimask` wrapper.
5. Reduce scratch allocations in `_optimask.py`.
6. Address memory-only issues in `nan_positions`, `scoring`, and time-series lag handling.

:notoc: true

Algorithm
#########

The `datafiller` library uses a model-based approach to impute missing values. This section provides an overview of the algorithm: how training data is selected for each missingness pattern, and where the `optimask` utility fits in.

The Core Idea
**************

For each column that contains missing values, `datafiller` treats that column as a target variable and the other columns as features. It then trains a machine learning model to predict the missing values based on the features that are available.

The key steps for imputing a single column are to identify the rows where the target column is missing, select training data where that target is present, train a model on that subset (by default a lightweight ridge regressor, ``FastRidge``, for numeric targets and a decision tree classifier for categorical ones), and then predict the missing values in the target column.

This process is repeated for each column that has missing data.

Per-Pattern Models
******************

The rows to impute rarely share the same set of observed features: one row may be missing only the target, another may also lack several feature columns. Rather than training a single model on the features common to all rows, `datafiller` groups the rows to impute by their pattern of observed features and trains one model per pattern, using exactly the features that pattern has available.

For each pattern, the training set is selected in two stages:

1. **Complete-case selection.** The rows that are fully observed on the pattern's feature columns (and on the target) are used directly. This is the common case, and it is computed cheaply from a per-column index of missing positions.
2. **`optimask` fallback.** When fewer than ``min_samples_train`` complete rows exist, the `optimask` algorithm searches for the largest possible "rectangular" subset of rows and columns that is free of missing values, trading some feature columns for more training rows.

How `optimask` works: it iteratively sorts rows and columns based on missing-value counts, a pareto-optimal strategy that pushes missing values toward the bottom-right of the matrix. After sorting, the problem becomes finding the largest rectangle of zeros in a binary matrix (where 1s represent missing values), which yields the largest complete subset of rows and columns for training.

Patterns that end up with identical training rows and columns share a single fitted model.

Making It Fast
**************

Fitting one model per missingness pattern could be expensive, so the default numeric path avoids refitting from scratch. Ridge regression depends on the training data only through its Gram matrix, so `datafiller` accumulates the Gram matrix of the augmented matrix ``[X, y, 1]`` once per column over the rows that are complete on **all** candidate features, then adds each pattern's few extra rows as a small correction and solves the ridge system directly. Custom regressors and categorical targets are fitted on the materialized training subset instead, since arbitrary estimators cannot be trained from summary statistics.

Performance-critical index manipulations are accelerated with Numba throughout.

By combining per-pattern models, complete-case selection, and the `optimask` fallback, `datafiller` can handle datasets with complex patterns of missingness and still produce reliable imputations.

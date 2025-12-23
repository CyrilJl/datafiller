:notoc: true

Algorithm
#########

The `datafiller` library uses a model-based approach to impute missing values. This section provides an overview of the algorithm, particularly the `optimask` utility that makes the imputation process robust.

The Core Idea
**************

For each column that contains missing values, `datafiller` treats that column as a target variable and the other columns as features. It then trains a machine learning model to predict the missing values based on the features that are available.

The key steps for imputing a single column are to identify the rows where the target column is missing, select training data where that target is present, train a regression model (for example, `LinearRegression`) on that subset, and then predict the missing values in the target column.

This process is repeated for each column that has missing data.

The `optimask` Algorithm
************************

A crucial part of the imputation process is selecting the best possible data for training the model. If the feature columns used for training also contain missing values, it can lead to poor model performance and inaccurate imputations.

This is where the `optimask` algorithm comes in. Before training a model for a specific target column, `optimask` is used to find the largest possible "rectangular" subset of the data that is free of missing values.

How it works: `optimask` iteratively sorts rows and columns based on missing-value counts, a pareto-optimal strategy that pushes missing values toward the bottom-right of the matrix. After sorting, the problem becomes finding the largest rectangle of zeros in a binary matrix (where 1s represent missing values), which yields the largest complete subset of rows and columns for training. That rectangle is used as the optimal training set, keeping model inputs clean and improving imputation quality.

By using `optimask`, `datafiller` can handle datasets with complex patterns of missingness and still produce reliable imputations.

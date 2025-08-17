Algorithm
#########

The `datafiller` library uses a model-based approach to impute missing values. This section provides an overview of the algorithm, particularly the `optimask` utility that makes the imputation process robust.

The Core Idea
**************

For each column that contains missing values, `datafiller` treats that column as a target variable and the other columns as features. It then trains a machine learning model to predict the missing values based on the features that are available.

The key steps for imputing a single column are:

1.  **Identify Missing Values**: Find the rows where the target column has missing values.
2.  **Select Training Data**: Select a subset of the data where the target column is *not* missing to use as training data.
3.  **Train a Model**: Train a regression model (e.g., `LinearRegression`) on the training data.
4.  **Predict Missing Values**: Use the trained model to predict the missing values in the target column.

This process is repeated for each column that has missing data.

The `optimask` Algorithm
************************

A crucial part of the imputation process is selecting the best possible data for training the model. If the feature columns used for training also contain missing values, it can lead to poor model performance and inaccurate imputations.

This is where the `optimask` algorithm comes in. Before training a model for a specific target column, `optimask` is used to find the largest possible "rectangular" subset of the data that is free of missing values.

How it works:

1.  **Pareto-Optimal Sorting**: `optimask` iteratively sorts the rows and columns based on the number of missing values they contain. This is a pareto-optimal sorting strategy that aims to push all the missing values towards the "bottom-right" of the matrix.
2.  **Largest Rectangle Problem**: After sorting, the problem is transformed into finding the largest rectangle of zeros in a binary matrix (where 1s represent missing values). This is a classic computer science problem that can be solved efficiently.
3.  **Optimal Training Set**: The resulting rectangle represents the largest, most complete subset of rows and columns that can be used for training. This ensures that the model is trained on high-quality data, leading to better imputation results.

By using `optimask`, `datafiller` can handle datasets with complex patterns of missingness and still produce reliable imputations.

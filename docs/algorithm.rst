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

Handling Categorical Data
*************************

When the input data is a pandas DataFrame, `datafiller` can also handle categorical columns (i.e., columns with `object` or `category` dtype). The approach is similar to the one for numerical data, but with a few key differences:

1.  **Separate Models**: Instead of a single `estimator`, the `MultivariateImputer` accepts a `regressor` for numerical columns and a `classifier` for categorical columns.
2.  **One-Hot Encoding**: When a column is being imputed, the categorical features used for prediction are transformed using one-hot encoding. This allows the model to use categorical information effectively.
3.  **Label Encoding for Target**: If the target column (the one being imputed) is categorical, its labels are encoded into integers before being passed to the classifier. The predicted integer labels are then transformed back to their original format.

This dual-model strategy allows `datafiller` to use the most appropriate technique for each type of data, leading to more accurate imputations in mixed-type datasets.

The `optimask` Algorithm
************************

A crucial part of the imputation process is selecting the best possible data for training the model. If the feature columns used for training also contain missing values, it can lead to poor model performance and inaccurate imputations.

This is where the `optimask` algorithm comes in. Before training a model for a specific target column, `optimask` is used to find the largest possible "rectangular" subset of the data that is free of missing values.

How it works:

1.  **Pareto-Optimal Sorting**: `optimask` iteratively sorts the rows and columns based on the number of missing values they contain. This is a pareto-optimal sorting strategy that aims to push all the missing values towards the "bottom-right" of the matrix.
2.  **Largest Rectangle Problem**: After sorting, the problem is transformed into finding the largest rectangle of zeros in a binary matrix (where 1s represent missing values). This is a classic computer science problem that can be solved efficiently.
3.  **Optimal Training Set**: The resulting rectangle represents the largest, most complete subset of rows and columns that can be used for training. This ensures that the model is trained on high-quality data, leading to better imputation results.

By using `optimask`, `datafiller` can handle datasets with complex patterns of missingness and still produce reliable imputations.

"""
    impute_zero(df::DataFrame; start_col::Int64 = 1, end_col::Int64 = size(df, 2))

Replaces missing elements in the specified columns with zero.

# Arguments

- `df`: dataframe with missing values.
- `start_col`: column index to start imputing from.
- `end_col`: column index to end imputing at.

# Example

```jldoctest
julia> df = DataFrame(A = [1, 2, 3],
                 B = [missing, missing, missing],
                 C = [missing, 4, 5],
                 D = [6, missing, 7],
                 E = [missing, missing, 10])
3×5 DataFrame
 Row │ A      B        C        D        E
     │ Int64  Missing  Int64?   Int64?   Int64?
─────┼───────────────────────────────────────────
   1 │     1  missing  missing        6  missing
   2 │     2  missing        4  missing  missing
   3 │     3  missing        5        7       10

julia> BigRiverJunbi.impute_zero(df)
3×5 DataFrame
 Row │ A       B       C       D       E      
     │ Int64?  Int64?  Int64?  Int64?  Int64? 
─────┼────────────────────────────────────────
   1 │      1       0       0       6       0
   2 │      2       0       4       0       0
   3 │      3       0       5       7      10
```
"""
function BigRiverJunbi.impute_zero(
        df::DataFrame; start_col::Int64 = 1, end_col::Int64 = size(df, 2)
    )
    m = Matrix(df[:, start_col:end_col])
    transformed = DataFrame(BigRiverJunbi.impute_zero!(m), Symbol.(names(df)[start_col:end_col]))
    return hcat(df[:, 1:(start_col - 1)], transformed, df[:, (end_col + 1):end])
end

"""
    impute_min(df::DataFrame; start_col::Int64 = 1, end_col::Int64 = size(df, 2))

Replaces missing elements in the specified columns with the minimum of non-missing elements
in the corresponding variable.

# Arguments

- `df`: dataframe with missing values.
- `start_col`: column index to start imputing from.
- `end_col`: column index to end imputing at.

# Example

```jldoctest
julia> df = DataFrame(A = [1, 2, 3],
                 B = [missing, missing, missing],
                 C = [missing, 4, 5],
                 D = [6, missing, 7],
                 E = [missing, missing, 10])
3×5 DataFrame
 Row │ A      B        C        D        E
     │ Int64  Missing  Int64?   Int64?   Int64?
─────┼───────────────────────────────────────────
   1 │     1  missing  missing        6  missing 
   2 │     2  missing        4  missing  missing 
   3 │     3  missing        5        7       10

julia> BigRiverJunbi.impute_min(df)
3×5 DataFrame
 Row │ A       B       C       D       E      
     │ Int64?  Int64?  Int64?  Int64?  Int64? 
─────┼────────────────────────────────────────
   1 │      1       1       1       6       1
   2 │      2       2       4       2       2
   3 │      3       3       5       7      10
```
"""
function BigRiverJunbi.impute_min(df::DataFrame; start_col::Int64 = 1, end_col::Int64 = size(df, 2))
    m = Matrix(df[:, start_col:end_col])
    transformed = DataFrame(BigRiverJunbi.impute_min!(m), Symbol.(names(df)[start_col:end_col]))
    return hcat(df[:, 1:(start_col - 1)], transformed, df[:, (end_col + 1):end])
end

"""
    impute_min_prob(df::DataFrame; q = 0.01, tune_sigma = 1, start_col::Int64 = 1, end_col::Int64 = size(df, 2))

Replaces missing values in the specified columns with random draws from a gaussian
distribution centered in the minimum value observed and with standard deviation equal to
the median value of the population of line-wise standard deviations.

# Arguments

- `df`: dataframe with missing values.
- `q`: quantile of the minimum values to use for imputation. Default is 0.01.
- `tune_sigma`: coefficient that controls the sd of the MNAR distribution:
                - 1 if the complete data distribution is supposed to be gaussian.
                - 0 < tune_sigma < 1 if the complete data distribution is supposed to be
                  left-censored.
               Default is 1.0.
- `start_col`: column index to start imputing from.
- `end_col`: column index to end imputing at.
"""
# TODO: add a example/doctest
function BigRiverJunbi.impute_min_prob(
        df::DataFrame; q = 0.01, tune_sigma = 1, start_col::Int64 = 1, end_col::Int64 = size(df, 2)
    )
    m = Matrix{Union{Missing, Float64}}(df[:, start_col:end_col])
    transformed = DataFrame(
        BigRiverJunbi.impute_min_prob!(m, q; tune_sigma), Symbol.(names(df)[start_col:end_col])
    )
    return hcat(df[:, 1:(start_col - 1)], transformed, df[:, (end_col + 1):end])
end

"""
    impute_half_min(df::DataFrame; start_col::Int64 = 1, end_col::Int64 = size(df, 2))

Replaces missing elements in the specified columns with half of the minimum of
non-missing elements in the corresponding variable.

# Arguments

- `df`: dataframe with missing values.
- `start_col`: column index to start imputing from.
- `end_col`: column index to end imputing at.

# Example

```jldoctest
julia> df = DataFrame(A = [1, 2, 3],
                 B = [missing, missing, missing],
                 C = [missing, 4, 5],
                 D = [6, missing, 7],
                 E = [missing, missing, 10])
3×5 DataFrame
 Row │ A      B        C        D        E
     │ Int64  Missing  Int64?   Int64?   Int64?
─────┼───────────────────────────────────────────
   1 │     1  missing  missing        6  missing
   2 │     2  missing        4  missing  missing
   3 │     3  missing        5        7       10

julia> BigRiverJunbi.impute_half_min(df)
3×5 DataFrame
 Row │ A       B       C       D       E      
     │ Int64?  Int64?  Int64?  Int64?  Int64? 
─────┼────────────────────────────────────────
   1 │      1       0       0       6       0
   2 │      2       1       4       1       1
   3 │      3       1       5       7      10
```
"""
function BigRiverJunbi.impute_half_min(
        df::DataFrame; start_col::Int64 = 1, end_col::Int64 = size(df, 2)
    )
    m = Matrix(df[:, start_col:end_col])
    transformed = DataFrame(
        BigRiverJunbi.impute_half_min!(m), Symbol.(names(df)[start_col:end_col])
    )
    return hcat(df[:, 1:(start_col - 1)], transformed, df[:, (end_col + 1):end])
end

"""
    impute_median_cat(df_missing::DataFrame; start_col::Int64 = 1, end_col::Int64 = size(df_missing, 2))

Returns imputed dataframe based on a categorical imputation:
    - 0: Missing values
    - 1: Values below the median
    - 2: Values equal to or above the median

# Arguments

- `df_missing`: dataframe with missing values.
- `start_col`: column index to start imputing from.
- `end_col`: column index to end imputing at.

# Example

```jldoctest
julia> df = DataFrame(A = [1, 2, 3],
                 B = [missing, missing, missing],
                 C = [missing, 4, 5],
                 D = [6, missing, 7],
                 E = [missing, missing, 10])
3×5 DataFrame
 Row │ A      B        C        D        E
     │ Int64  Missing  Int64?   Int64?   Int64?
─────┼───────────────────────────────────────────
   1 │     1  missing  missing        6  missing
   2 │     2  missing        4  missing  missing
   3 │     3  missing        5        7       10

julia> BigRiverJunbi.impute_median_cat(df)
3×5 DataFrame
 Row │ A       B       C       D       E      
     │ Int64?  Int64?  Int64?  Int64?  Int64? 
─────┼────────────────────────────────────────
   1 │      1       0       0       1       0
   2 │      2       0       1       0       0
   3 │      2       0       2       2       2
```
"""
function BigRiverJunbi.impute_median_cat(
        df_missing::DataFrame; start_col::Int64 = 1, end_col::Int64 = size(df_missing, 2)
    )
    m = Matrix(df_missing[:, start_col:end_col])
    transformed = DataFrame(
        BigRiverJunbi.impute_median_cat!(m), Symbol.(names(df_missing)[start_col:end_col])
    )
    return hcat(df_missing[:, 1:(start_col - 1)], transformed, df_missing[:, (end_col + 1):end])
end

"""
    imputeKNN(df::DataFrame; k = 5, threshold = 0.2, start_col = 1, end_col = size(df, 2))

Replaces missing elements based on k-nearest neighbors (KNN) imputation.

# Arguments

- `df`: dataframe with missing values.
- `k`: number of nearest neighbors to use for imputation.
- `threshold`: threshold for the number of missing neighbors above which the imputation is
  skipped.
- `start_col`: column index to start imputing from.
- `end_col`: column index to end imputing at.
"""
function BigRiverJunbi.imputeKNN(
        df::DataFrame;
        k::Int = 5,
        threshold::Float64 = 0.2,
        start_col::Int64 = 1,
        end_col::Int64 = size(df, 2)
    )
    # TODO: add a example/doctest
    mat = Matrix{Union{Missing, Float64}}(df[:, start_col:end_col])
    transformed = DataFrame(
        BigRiverJunbi.imputeKNN(mat, k; threshold, dims = 1), Symbol.(names(df)[start_col:end_col])
    )
    return hcat(df[:, 1:(start_col - 1)], transformed, df[:, (end_col + 1):end])
end

"""
    impute_QRILC(df::DataFrame; start_col::Int64 = 1, end_col::Int64 = size(df, 2))

Replaces missing elements in the specified columns based on the "Quantile regression
Imputation for left-censored data" (QRILC) method.

# Arguments

- `df`: dataframe with missing values.
- `start_col`: column index to start imputing from.
- `end_col`: column index to end imputing at.
"""
# TODO: add a example/doctest
function BigRiverJunbi.impute_QRILC(
        df::DataFrame; start_col::Int64 = 1, end_col::Int64 = size(df, 2)
    )
    mat = Matrix{Union{Missing, Float64}}(df[:, start_col:end_col])
    transformed = DataFrame(BigRiverJunbi.impute_QRILC!(mat), Symbol.(names(df)[start_col:end_col]))
    return hcat(df[:, 1:(start_col - 1)], transformed, df[:, (end_col + 1):end])
end

# TODO: add docstring
# TODO: add a example/doctest
function BigRiverJunbi.imputeSVD(df::DataFrame; start_col::Int64 = 1, end_col::Int64 = size(df, 2))
    mat = Matrix{Union{Missing, Float64}}(df[:, start_col:end_col])
    transformed = DataFrame(BigRiverJunbi.imputeSVD(mat), Symbol.(names(df)[start_col:end_col]))
    return hcat(df[:, 1:(start_col - 1)], transformed, df[:, (end_col + 1):end])
end

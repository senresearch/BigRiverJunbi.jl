"""
    substitute(
        data::AbstractArray{Union{Missing, Float64}},
        statistic::Function;
        dims::Union{Nothing, Int} = nothing
    )

Substitutes missing values with the value calculated by the statistic function along the
specified dimension and returns a new array without modifying the original array.

# Arguments

- `data`: array of values. One example: matrix of metabolomics data, where the rows are
  the features and the columns are the samples.
- `statistic`: function that calculates the value to substitute the missing values.
- `dims`: dimension along which the statistic is calculated.
"""
function substitute(
        data::AbstractArray{Union{Missing, Float64}},
        statistic::Function;
        dims::Union{Nothing, Int} = nothing
)
    substitute!(trycopy(data), statistic; dims)
end

"""
    substitute!(
        data::AbstractArray{Union{Missing, Float64}},
        statistic::Function;
        dims::Union{Nothing, Int} = nothing
    )

Substitutes missing values with the value calculated by the statistic function along the
specified dimension and modifies the original array in place.

# Arguments

- `data`: array of values. One example: matrix of metabolomics data, where the rows are the
  features and the columns are the samples.
- `statistic`: function that calculates the value to substitute the missing values.
- `dims`: dimension along which the statistic is calculated.
"""
function substitute!(
        data::AbstractArray{Union{Missing, Float64}},
        statistic::Function;
        dims::Union{Nothing, Int} = nothing
)
    # if dims is nothing, substitute the whole array
    isnothing(dims) && return _substitute!(data, statistic)
    # if dims is specified, it must be smaller than the dimension of the array
    dims > ndims(data) && error("dims must be smaller than the dimension of the array.")
    # iterate over the slices of the array along the specified dimension
    for slice in eachslice(data; dims)
        _substitute!(slice, statistic)
    end
    return data
end

function _substitute!(data::AbstractArray{Union{Missing, Float64}}, statistic::Function)
    # get the mask of the non-missing values
    mask = .!ismissing.(data)
    # substitute the missing values with the value calculated by the statistic
    if any(mask)
        x = statistic(disallowmissing(data[mask]))
        replace!(data, missing => x)
    else
        error(
            "All values in the input slice are missing. This usually happens when " *
            "there is a row or column with all missing values along the specified " *
            "dimension. Please check your data."
        )
    end
    return data
end

"""
    impute_zero(df::DataFrame; start_col::Int64 = 1)

Replaces missing elements in the specified columns with zero.

# Arguments

- `df`: dataframe with missing values.
- `start_col`: column index to start imputing from.

# Examples

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
 Row │ A         B         C         D         E
     │ Float64?  Float64?  Float64?  Float64?  Float64?
─────┼──────────────────────────────────────────────────
   1 │      1.0       0.0       0.0       6.0       0.0
   2 │      2.0       0.0       4.0       0.0       0.0
   3 │      3.0       0.0       5.0       7.0      10.0
```
"""
function impute_zero(df::DataFrame; start_col::Int64 = 1)
    m = Matrix{Union{Missing, Float64}}(df[:, start_col:end])
    m = impute_zero(m)
    return DataFrame(m, Symbol.(names(df)[start_col:end]))
end

"""
    impute_zero(data::Matrix{Union{Missing, Float64}})

Returns a matrix with missing elements replaced with zero without modifying the original
matrix.

# Arguments

- `data`: matrix of omics value, e.g., metabolomics matrix, where the rows are the samples
        and the columns are the features.
"""
impute_zero(data::Matrix{Union{Missing, Float64}}) = impute_zero!(trycopy(data))

"""
    impute_zero!(data::Matrix{Union{Missing, Float64}})

Modifies the original matrix in place to replace missing elements with zero.

# Arguments

- `data`: matrix of omics value, e.g., metabolomics matrix, where the rows are the samples
  and the columns are the features.
"""
impute_zero!(data::Matrix{Union{Missing, Float64}}) = substitute!(data, x -> 0; dims = 1)

"""
    impute_min(df::DataFrame; start_col::Int64 = 1)

Replaces missing elements in the specified columns with the minimum of non-missing elements
in the corresponding variable.

# Arguments

- `df`: dataframe with missing values.
- `start_col`: column index to start imputing from.

# Examples

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
 Row │ A         B         C         D         E
     │ Float64?  Float64?  Float64?  Float64?  Float64? 
─────┼──────────────────────────────────────────────────
   1 │      1.0       1.0       1.0       6.0       1.0
   2 │      2.0       2.0       4.0       2.0       2.0
   3 │      3.0       3.0       5.0       7.0      10.0
```
"""
function impute_min(df::DataFrame; start_col::Int64 = 1)
    m = Matrix{Union{Missing, Float64}}(df[:, start_col:end])
    m = impute_min(m)
    return DataFrame(m, Symbol.(names(df)[start_col:end]))
end
impute_min(data::Matrix{Union{Missing, Float64}}) = impute_min!(trycopy(data))
impute_min!(data::Matrix{Union{Missing, Float64}}) = substitute!(data, minimum; dims = 1)

"""
    impute_min_prob(df::DataFrame; start_col::Int64 = 1, q = 0.01; tune_sigma = 1)

Replaces missing values in the specified columns with random draws from a gaussian
distribution centered in the minimum value observed and with standard deviation equal to
the median value of the population of line-wise standard deviations.

# Arguments

- `df`: dataframe with missing values.
- `start_col`: column index to start imputing from.
- `q`: quantile of the minimum values to use for imputation. Default is 0.01.
- `tune_sigma`: coefficient that controls the sd of the MNAR distribution:
                - 1 if the complete data distribution is supposed to be gaussian.
                - 0 < tune_sigma < 1 if the complete data distribution is supposed to be
                  left-censored.
               Default is 1.0.
"""
function impute_min_prob(df::DataFrame; start_col::Int64 = 1, q = 0.01, tune_sigma = 1)
    m = Matrix{Union{Missing, Float64}}(df[:, start_col:end])
    m = impute_min_prob(m, q; tune_sigma)
    return DataFrame(m, Symbol.(names(df)[start_col:end]))
end

"""
    impute_min_prob(data::Matrix{Union{Missing, Float64}}, q = 0.01; tune_sigma = 1)

Replaces missing values with random draws from a gaussian distribution centered in the
minimum value observed and with standard deviation equal to the median value of the
population of line-wise standard deviations. Returns a new matrix without modifying the
original matrix.

# Arguments

- `data`: matrix of omics value, e.g., metabolomics matrix, where the rows are the samples
  and the columns are the features.
- `q`: quantile of the minimum values to use for imputation. Default is 0.01.
- `tune_sigma`: coefficient that controls the sd of the MNAR distribution:
                - 1 if the complete data distribution is supposed to be gaussian.
                - 0 < tune_sigma < 1 if the complete data distribution is supposed to be
                  left-censored.
               Default is 1.0.
"""
function impute_min_prob(data::Matrix{Union{Missing, Float64}}, q = 0.01; tune_sigma = 1)
    impute_min_prob!(trycopy(data), q; tune_sigma)
end

"""
    impute_min_prob!(data::Matrix{Union{Missing, Float64}}, q = 0.01; tune_sigma = 1)

Replaces missing values with random draws from a gaussian distribution centered in the
minimum value observed and with standard deviation equal to the median value of the
population of line-wise standard deviations. Modifies the original matrix in place.

# Arguments

- `data`: matrix of omics value, e.g., metabolomics matrix, where the rows are the samples
  and the columns are the features.
- `q`: quantile of the minimum values to use for imputation. Default is 0.01.
- `tune_sigma`: coefficient that controls the sd of the MNAR distribution:
                - 1 if the complete data distribution is supposed to be gaussian.
                - 0 < tune_sigma < 1 if the complete data distribution is supposed to be
                  left-censored.
               Default is 1.0.
"""
function impute_min_prob!(data::Matrix{Union{Missing, Float64}}, q = 0.01; tune_sigma = 1)
    n_samples, n_features = size(data)
    # select the minimum values sample-wise (corresponding to the q-th quantile)
    min_vals = mapslices(data, dims = 1) do x
        quantile(skipmissing(x), q)
    end
    # estimate protein-wise standard deviation using only proteins containing more than 50% non-NAs
    pNA_cols = sum(!ismissing, data; dims = 1) ./ n_samples
    pNA_cols_mask = vec(pNA_cols .> 0.5)
    data_filtered = data[:, pNA_cols_mask]
    sd_vals = mapslices(data_filtered; dims = 1) do x
        std(skipmissing(x))
    end
    sd_temp = median(sd_vals) * tune_sigma
    # generate data from a normal distribution with the estimated parameters
    for i in 1:n_samples
        dist = Normal(min_vals[i], sd_temp)
        curr_sample = data[:, i]
        curr_sample_imputed = trycopy(curr_sample)
        missing_idx = findall(ismissing, curr_sample)
        curr_sample_imputed[missing_idx] .= rand(dist, n_features)[missing_idx]
        data[:, i] = curr_sample_imputed
    end
    return data
end

"""
    impute_half_min(df::DataFrame; start_col::Int64 = 1)

Replaces missing elements in the specified columns with half of the minimum of
non-missing elements in the corresponding variable.

# Arguments

- `df`: dataframe with missing values.
- `start_col`: column index to start imputing from.

# Examples

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
 Row │ A         B         C         D         E
     │ Float64?  Float64?  Float64?  Float64?  Float64?
─────┼──────────────────────────────────────────────────
   1 │      1.0       0.5       0.5       6.0       0.5
   2 │      2.0       1.0       4.0       1.0       1.0
   3 │      3.0       1.5       5.0       7.0      10.0
```
"""
function impute_half_min(df::DataFrame; start_col::Int64 = 1)
    m = Matrix{Union{Missing, Float64}}(df[:, start_col:end])
    m = impute_half_min(m)
    return DataFrame(m, Symbol.(names(df)[start_col:end]))
end

function impute_half_min(m::Matrix{Union{Missing, Float64}})
    substitute(m, x -> minimum(x) / 2; dims = 1)
end

"""
    impute_cat(df_missing::DataFrame; start_col::Int64 = 1)

Returns imputated dataframe based on a categorical imputation:
    - 0: Missing values
    - 1: Values below the median
    - 2: Values equal to or above the median

# Arguments

- `df_missing`: dataframe with missing values.
- `start_col`: column index to start imputing from.

# Examples

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

julia> BigRiverJunbi.impute_cat(df)
3×5 DataFrame
 Row │ A         B         C         D         E
     │ Float64?  Float64?  Float64?  Float64?  Float64?
─────┼──────────────────────────────────────────────────
   1 │      1.0       0.0       0.0       1.0       0.0
   2 │      2.0       0.0       1.0       0.0       0.0
   3 │      2.0       0.0       2.0       2.0       2.0
```
"""
function impute_cat(df_missing::DataFrame; start_col::Int64 = 1)
    m = Matrix{Union{Missing, Float64}}(df_missing[:, start_col:end])
    m = impute_cat(m)
    return DataFrame(m, Symbol.(names(df_missing)[start_col:end]))
end

"""
    impute_cat(data::Matrix{Union{Missing, Float64}})

Imputes missing elements based on a categorical imputation:
    - 0: Missing values
    - 1: Values below the median
    - 2: Values equal to or above the median
Returns a new matrix without modifying the original matrix.

# Arguments

- `data`: matrix of omics value, e.g., metabolomics matrix, where the rows are the samples
  and the columns are the features.
"""
impute_cat(data::Matrix{Union{Missing, Float64}}) = impute_cat!(trycopy(data))

"""
    impute_cat!(data::Matrix{Union{Missing, Float64}})

Imputes missing elements based on a categorical imputation:
    - 0: Missing values
    - 1: Values below the median
    - 2: Values equal to or above the median
Modifies the original matrix in place.

# Arguments

- `data`: matrix of omics value, e.g., metabolomics matrix, where the rows are the samples
  and the columns are the features.
"""
function impute_cat!(data::Matrix{Union{Missing, Float64}})
    @views for i in axes(data, 2)
        # if there are only missing values, skip
        if all(ismissing, data[:, i])
            continue
        end
        # find indexes of non missing for each column
        idx_not_missing = findall(.!ismissing.(data[:, i]))
        med = median(data[idx_not_missing, i])
        # replace values below and above the median by 1 and 2 respectively
        replace!(data[idx_not_missing, i]) do x
            x < med ? 1 : 2
        end
    end
    # replace values with missing by 0
    replace!(data, missing => 0)
    return data
end

"""
    imputeKNN(df::DataFrame; k = 5, threshold = 0.2, start_col = 1)

Replaces missing elements based on k-nearest neighbors (KNN) imputation.

# Arguments

- `df`: dataframe with missing values.
- `k`: number of nearest neighbors to use for imputation.
- `threshold`: threshold for the number of missing neighbors.
"""
function imputeKNN(
        df::DataFrame;
        k::Int = 5,
        threshold::Float64 = 0.2,
        start_col::Int64 = 1
)
    mat = Matrix{Union{Missing, Float64}}(df[:, start_col:end])
    return DataFrame(
        imputeKNN(mat, k; threshold, dims = 1), Symbol.(names(df)[start_col:end])
    )
end

"""
    imputeKNN(
        data::AbstractMatrix{Union{Missing, Float64}},
        k::Int = 1;
        threshold::Float64 = 0.5,
        dims::Union{Nothing, Int} = nothing,
        distance::M = Euclidean()
    ) where {M <: NearestNeighbors.MinkowskiMetric}

Replaces missing elements based on k-nearest neighbors (KNN). Returns a new matrix
without modifying the original matrix. This method is almost an exact copy of the KNN
imputation method from [Impute.jl](https://github.com/invenia/Impute.jl).

# Arguments

- `data`: matrix of omics value, e.g., metabolomics matrix, where the rows are the samples
  and the columns are the features.
- `k`: number of nearest neighbors to use for imputation.
- `threshold`: threshold for the number of missing neighbors.
- `dims`: dimension along which the statistic is calculated.
- `distance`: distance metric to use for the nearest neighbors search, taken from
  Distances.jl. Default is `Euclidean()`. This can only be one of the
  Minkowski metrics i.e. Euclidean, Cityblock, Minkowski and Chebyshev.
"""
# TODO: add a example/doctest
function imputeKNN(
        data::AbstractMatrix{Union{Missing, Float64}},
        k::Int = 1;
        threshold::Float64 = 0.5,
        dims::Union{Nothing, Int} = nothing,
        distance::M = Euclidean()
) where {M <: NearestNeighbors.MinkowskiMetric}
    # check arguments
    k < 1 &&
        throw(ArgumentError("The number of nearset neighbors should be greater than 0"))
    !(0 < threshold < 1) &&
        throw(ArgumentError("Missing neighbors threshold should be within 0 to 1"))

    return imputeKNN!(trycopy(data), k + 1, threshold, dims, distance)
end

"""
    imputeKNN!(
        data::AbstractMatrix{Union{Missing, Float64}},
        k::Int,
        threshold::Float64,
        dims::Union{Nothing, Int},
        distance::M
    ) where {M <: NearestNeighbors.MinkowskiMetric}

Replaces missing elements based on k-nearest neighbors (KNN) imputation. Modifies the
original matrix in place. This method is almost an exact copy of the KNN imputation
method from [Impute.jl](https://github.com/invenia/Impute.jl).

# Arguments

- `data`: matrix of omics value, e.g., metabolomics matrix, where the rows are the samples
  and the columns are the features.
- `k`: number of nearest neighbors to use for imputation.
- `threshold`: threshold for the number of missing neighbors.
- `dims`: dimension along which the statistic is calculated.
- `distance`: distance metric to use for the nearest neighbors search, taken from
  Distances.jl. Default is `Euclidean()`. This can only be one of the
  Minkowski metrics i.e. Euclidean, Cityblock, Minkowski and Chebyshev.
"""
function imputeKNN!(
        data::AbstractMatrix{Union{Missing, Float64}},
        k::Int,
        threshold::Float64,
        dims::Union{Nothing, Int},
        distance::M
) where {M <: NearestNeighbors.MinkowskiMetric}
    # KDTree wants dims x nrows, so we transpose the data
    X = dims == 1 ? data : transpose(data)
    # get mask array
    missing_mask = ismissing.(X)
    # impute missing values as mean
    X = substitute!(X, mean; dims = 1)
    # disallow missing values
    X = disallowmissing(X)
    # search points are observations containing missing values
    points = X[:, vec(any(missing_mask; dims = 1))]
    # construct KDTree
    kdtree = KDTree(X, distance)

    # query for neighbors to missing observations
    for (idxs, dists) in zip(NearestNeighbors.knn(kdtree, points, k, true)...)
        # closest neighbor should always be input data point (distance of zero)
        @assert iszero(first(dists))
        # Location of point to impute
        j = first(idxs)
        # Update each missing value in this point
        for i in axes(points, 1)
            # Skip non-missing elements
            missing_mask[i, j] || continue
            # grab neighbor mask to excluding neighbor values that were also missing
            neighbor_mask = missing_mask[i, idxs]
            # skip if there are too many missing neighbor values
            (count(neighbor_mask) / k) > threshold && continue
            # weight valid neighbors based on inverse distance
            neighbor_dists = dists[.!neighbor_mask]
            # weights are inverse of distances
            wv = 1.0 ./ neighbor_dists
            wv_sum = sum(wv)
            # fill with the weighted mean of neighbors if the sum of the weights
            # are non-zero and finite
            if isfinite(wv_sum) && !iszero(wv_sum)
                neighbor_vals = X[i, idxs[.!neighbor_mask]]
                X[i, j] = mean(neighbor_vals, Weights(wv, wv_sum))
            end
        end
    end

    # for type stability
    return allowmissing(dims == 1 ? X : transpose(X))
end

function imputeSVD(df::DataFrame; start_col::Int64 = 1)
    mat = Matrix{Union{Missing, Float64}}(df[:, start_col:end])
    return DataFrame(imputeSVD(mat), Symbol.(names(df)[start_col:end]))
end

function imputeSVD(
        data::AbstractMatrix{Union{Missing, Float64}};
        rank::Union{Nothing, Int} = nothing,
        tol::Float64 = 1e-10,
        maxiter::Int = 100,
        limits::Union{Tuple{Float64, Float64}, Nothing} = nothing,
        dims::Union{Nothing, Int} = nothing,
        verbose::Bool = true
)
    return imputeSVD!(trycopy(data); rank, tol, maxiter, limits, dims, verbose)
end

function imputeSVD!(
        data::AbstractMatrix{Union{Missing, Float64}};
        rank::Union{Nothing, Int},
        tol::Float64,
        maxiter::Int,
        limits::Union{Tuple{Float64, Float64}, Nothing},
        dims::Union{Nothing, Int},
        verbose::Bool
)
    n, p = size(data)
    k = isnothing(rank) ? 0 : min(rank, p - 1)
    S = zeros(min(n, p))
    X = zeros(n, p)

    # Get our before and after views of our missing and non-missing data
    mmask = ismissing.(data)
    omask = .!mmask
    mdata = data[mmask]
    mX = X[mmask]
    odata = data[omask]
    oX = X[omask]

    # substitute missing values with median
    substitute!(data, median; dims)
    # print debug information
    C = sum(abs2, mdata - mX) / sum(abs2, mdata)
    err = mean(abs.(odata - oX))
    verbose && @debug("Before", Diff=sum(mdata-mX), MAE=err, convergence=C,
        normsq=sum(abs2, mdata), mX[1])

    for i in 1:maxiter
        isnothing(rank) && (k = min(k + 1, p - 1, n - 1))
        # Compute the SVD and produce a low-rank approximation of the data
        F = svd(data)
        S[1:k] .= F.S[1:k]
        X = F.U * Diagonal(S) * F.Vt
        # Clamp the values if necessary
        !isnothing(limits) && clamp!(X, limits...)
        # Test for convergence
        mdata = data[mmask]
        mX = X[mmask]
        odata = data[omask]
        oX = X[omask]
        err = mean(abs.(odata - oX))
        C = sum(abs2, mdata - mX) / sum(abs2, mdata)
        # Print the error between reconstruction and observed inputs
        verbose && @debug("Iteration", i, Diff=sum(mdata-mX), MAE=err,
            MSS=sum(abs2, mdata), convergence=C)
        # Update missing values
        data[mmask] .= X[mmask]
        isfinite(C) && C < tol && break
    end

    return data
end

"""
    impute_QRILC(df::DataFrame; start_col::Int64 = 1)

Replaces missing elements in the specified columns based on the "Quantile regression
Imputation for left-censored data" (QRILC) method.

# Arguments

- `df`: dataframe with missing values.
- `start_col`: column index to start imputing from.
"""
# TODO: add a example/doctest
function impute_QRILC(df::DataFrame; start_col::Int64 = 1)
    mat = Matrix{Union{Missing, Float64}}(df[:, start_col:end])
    mat = impute_QRILC(mat)
    return DataFrame(mat, Symbol.(names(df)[start_col:end]))
end

"""
    impute_QRILC(
        data::Matrix{Union{Missing, Float64}};
        tune_sigma::Float64 = 1.0,
        eps::Float64 = 0.005
    )

Returns imputated matrix based on the "Quantile regression Imputation for left-censored
data" (QRILC) method. The function is based on the function `impute.QRILC` from the
`imputeLCMD.R` package, with one difference: the default value of `eps` is set to 0.005
instead of 0.001.

# Arguments

- `data`: matrix of omics value, e.g., metabolomics matrix, where the rows are the samples
  and the columns are the features.
- `tune_sigma`: coefficient that controls the sd of the MNAR distribution:
                - 1 if the complete data distribution is supposed to be gaussian.
                - 0 < tune_sigma < 1 if the complete data distribution is supposed to be
                  left-censored.
  Default is 1.0.
- `eps`: small value added to the quantile for stability.
"""
function impute_QRILC(
        data::Matrix{Union{Missing, Float64}};
        tune_sigma::Float64 = 1.0,
        eps::Float64 = 0.005
)
    impute_QRILC!(trycopy(data); tune_sigma, eps)
end

"""
    impute_QRILC!(
        data::Matrix{Union{Missing, Float64}};
        tune_sigma::Float64 = 1.0,
        eps::Float64 = 0.005
    )

Imputes missing elements based on the "Quantile regression Imputation for left-censored
data" (QRILC) method. Modifies the original matrix in place. The function is based on
the function `impute.QRILC` from the `imputeLCMD.R` package, with one difference: the
default value of `eps` is set to 0.005 instead of 0.001.

# Arguments

- `data`: matrix of omics value, e.g., metabolomics matrix, where the rows are the samples
  and the columns are the features.
- `tune_sigma`: coefficient that controls the sd of the MNAR distribution:
                - 1 if the complete data distribution is supposed to be gaussian.
                - 0 < tune_sigma < 1 if the complete data distribution is supposed to be
                  left-censored.
- `eps`: small value added to the quantile for stability.
"""
# TODO: elaborate on eps and why it is set to 0.005
function impute_QRILC!(
        data::Matrix{Union{Missing, Float64}};
        tune_sigma::Float64 = 1.0,
        eps::Float64 = 0.005
)
    # Get dimensions of the data
    n_samples, n_features = size(data)
    for i in 1:n_samples
        curr_sample = data[:, i]
        # Calculate the percentage of missing values
        pNAs = count(ismissing, curr_sample) / length(curr_sample)
        # Estimate the mean and standard deviation of the original distribution using quantile regression
        upper_q = 0.99
        q_normal = quantile(Normal(0, 1), LinRange(pNAs + eps, upper_q + eps, 100))
        q_curr_sample = quantile(
            skipmissing(curr_sample), LinRange(eps, upper_q + eps, 100)
        )
        temp_QR = lm(
            hcat(ones(length(q_normal), 1), reshape(q_normal, :, 1)), q_curr_sample
        )
        # get the coefficients of the quantile regression
        coefs = coef(temp_QR)
        mean_CDD, sd_CDD = coefs[1], abs(coefs[2])
        # generate data from a truncated normal distribution with the estimated parameters
        truncated_dist = truncated(
            Normal(mean_CDD, sd_CDD * tune_sigma);
            upper = quantile(Normal(mean_CDD, sd_CDD), pNAs + eps)
        )
        curr_sample_imputed = trycopy(curr_sample)
        missing_idx = findall(ismissing, curr_sample)
        curr_sample_imputed[missing_idx] .= rand(truncated_dist, n_features)[missing_idx]
        data[:, i] = curr_sample_imputed
    end
    return data
end

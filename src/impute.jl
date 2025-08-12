"""
    substitute(
        data::AbstractArray{<:Union{Missing, Real}},
        statistic::Function;
        dims::Union{Nothing, Int} = nothing
    )

Substitutes missing values with the value calculated by the statistic function along the
specified dimension and returns a new array without modifying the original array.

# Arguments

- `data`: array of values. One example: matrix of metabolomics data, where the rows are
  the features and the columns are the samples.
- `statistic`: function that calculates the value to substitute the missing values. The
  function must return a value of the same type as the data.
- `dims`: dimension along which the statistic is calculated.
"""
function substitute(
        data::AbstractArray{<:Union{Missing, Real}},
        statistic::Function;
        dims::Union{Nothing, Int} = nothing
    )
    return substitute!(trycopy(data), statistic; dims)
end

"""
    substitute!(
        data::AbstractArray{<:Union{Missing, Real}},
        statistic::Function;
        dims::Union{Nothing, Int} = nothing
    )

Substitutes missing values with the value calculated by the statistic function along the
specified dimension and writes the result back to the original array.

# Arguments

- `data`: array of values. One example: matrix of metabolomics data, where the rows are the
  features and the columns are the samples.
- `statistic`: function that calculates the value to substitute the missing values. The
  function must return a value of the same type as the data.
- `dims`: dimension along which the statistic is calculated.
"""
function substitute!(
        data::AbstractArray{<:Union{Missing, Real}},
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

function _substitute!(data::AbstractArray{<:Union{Missing, Real}}, statistic::Function)
    # get the mask of the non-missing values
    mask = .!ismissing.(data)
    # substitute the missing values with the value calculated by the statistic
    if any(mask)
        x = statistic(disallowmissing(data[mask]))
        try
            replace!(data, missing => x)
        catch
            error(
                "Failed to replace missing values with the value calculated by " *
                    "the statistic. This usually happens when the type returned by " *
                    "the statistic function is not the same as the type of the data. " *
                    "Please check your statistic function or promote the type of the " *
                    "data to the type returned by the statistic function and try again."
            )
        end
        return data
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
    impute_zero(data::Matrix{<:Union{Missing, Real}})

Returns a matrix with missing elements replaced with zero without modifying the original
matrix.

# Arguments

- `data`: matrix of omics value, e.g., metabolomics matrix, where the rows are the samples
        and the columns are the features.
"""
impute_zero(data::Matrix{<:Union{Missing, Real}}) = impute_zero!(trycopy(data))

"""
    impute_zero!(data::Matrix{<:Union{Missing, Real}})

Replace missing elements with zero and writes the result back to the original matrix.

# Arguments

- `data`: matrix of omics value, e.g., metabolomics matrix, where the rows are the samples
  and the columns are the features.
"""
impute_zero!(data::Matrix{<:Union{Missing, Real}}) = substitute!(data, x -> 0)

"""
    impute_min(data::Matrix{<:Union{Missing, Real}}; dims::Union{Nothing, Int} = nothing)

Replaces missing elements with the minimum value of the non-missing elements and returns a
new matrix without modifying the original matrix.

# Arguments

- `data`: matrix of omics value, e.g., metabolomics matrix, where the rows are the samples
  and the columns are the features.
- `dims`: dimension along which the minimum values are calculated. Default is nothing, which
  means the whole matrix is used.
"""
function impute_min(data::Matrix{<:Union{Missing, Real}}; dims::Union{Nothing, Int} = nothing)
    return impute_min!(trycopy(data); dims)
end

"""
    impute_min!(data::Matrix{<:Union{Missing, Real}}; dims::Union{Nothing, Int} = nothing)

Replaces missing elements with the minimum value of the non-missing elements and writes the
result back to the original matrix.

# Arguments

- `data`: matrix of omics value, e.g., metabolomics matrix, where the rows are the samples
  and the columns are the features.
- `dims`: dimension along which the minimum values are calculated. Default is nothing, which
  means the whole matrix is used.
"""
function impute_min!(data::Matrix{<:Union{Missing, Real}}; dims::Union{Nothing, Int} = nothing)
    return substitute!(data, minimum; dims)
end

"""
    impute_half_min(data::Matrix{<:Union{Missing, Real}}; dims::Union{Nothing, Int} = nothing)

Replaces missing elements with half of the minimum value of the non-missing elements and
returns a new matrix without modifying the original matrix.

!!! note
    For integer matrices, the half of the minimum value is calculated by integer division i.e.
    the result of the division is rounded down to the nearest integer if the result is not
    an integer.

# Arguments

- `data`: matrix of omics value, e.g., metabolomics matrix, where the rows are the samples
  and the columns are the features.
- `dims`: dimension along which the minimum values are calculated. Default is nothing, which
  means the whole matrix is used.
"""
function impute_half_min(data::Matrix{<:Union{Missing, Real}}; dims::Union{Nothing, Int} = nothing)
    return impute_half_min!(trycopy(data); dims)
end

"""
    impute_half_min!(data::Matrix{<:Union{Missing, Real}}; dims::Union{Nothing, Int} = nothing)

Replaces missing elements with half of the minimum value of the non-missing elements and
writes the result back to the original matrix.

!!! note
    For integer matrices, the half of the minimum value is calculated by integer division
    i.e. the result of the division is rounded down to the nearest integer if the result is
    not an integer.

# Arguments

- `data`: matrix of omics value, e.g., metabolomics matrix, where the rows are the samples
  and the columns are the features.
- `dims`: dimension along which the minimum values are calculated. Default is nothing, which
  means the whole matrix is used.
"""
function impute_half_min!(data::Matrix{<:Union{Missing, Real}}; dims::Union{Nothing, Int} = nothing)
    return substitute!(data, x -> minimum(x) / 2; dims)
end

function impute_half_min!(
        data::Matrix{<:Union{Missing, Integer}}; dims::Union{Nothing, Int} = nothing
    )
    return substitute!(data, x -> minimum(x) ÷ 2; dims)
end

"""
    impute_min_prob(
        data::Matrix{<:Union{Missing, Real}}, q::Float64 = 0.01;
        tune_sigma::Float64 = 1.0, dims::Int = 1,
        rng::AbstractRNG = Random.default_rng()
    )

Replaces missing values with random draws from a gaussian distribution centered in the
minimum value observed and with standard deviation equal to the median value of the
population of line-wise standard deviations. Returns a new matrix without modifying the
original matrix.

# Arguments

- `data`: matrix of omics value, e.g., metabolomics matrix, where the rows are the samples
  and the columns are the features.
- `q`: quantile of the minimum values to use for imputation. Default is 0.01.
- `dims`: dimension along which the minimum values are calculated. Default is 1.
- `tune_sigma`: coefficient that controls the sd of the MNAR distribution:
                - 1 if the complete data distribution is supposed to be gaussian.
                - 0 < tune_sigma < 1 if the complete data distribution is supposed to be
                  left-censored.
               Default is 1.0.
"""
function impute_min_prob(
        data::Matrix{<:Union{Missing, Real}}, q::Float64 = 0.01;
        tune_sigma::Float64 = 1.0, dims::Int = 1,
        rng::AbstractRNG = Random.default_rng()
    )
    promoted = convert(Matrix{Union{Missing, Float64}}, data)
    return impute_min_prob!(trycopy(promoted), q; tune_sigma, dims, rng)
end

"""
    impute_min_prob!(
        data::Matrix{Union{Missing, Float64}}, q = 0.01;
        tune_sigma = 1, dims::Int = 1,
        rng::AbstractRNG = Random.default_rng()
    )

Replaces missing values with random draws from a gaussian distribution centered around the
minimum value observed and with standard deviation equal to the median value of the
population of line-wise standard deviations. Writes the result back to the original matrix.

# Arguments

- `data`: matrix of omics value, e.g., metabolomics matrix, where the rows are the samples
  and the columns are the features.
- `q`: quantile of the minimum values to use for imputation. Default is 0.01.
- `dims`: dimension along which the minimum values are calculated. Default is 1.
- `tune_sigma`: coefficient that controls the sd of the MNAR distribution:
                - 1 if the complete data distribution is supposed to be gaussian.
                - 0 < tune_sigma < 1 if the complete data distribution is supposed to be
                  left-censored.
                Default is 1.0.
"""
function impute_min_prob!(
        data::Matrix{Union{Missing, Float64}}, q = 0.01;
        tune_sigma = 1, dims::Int = 1,
        rng::AbstractRNG = Random.default_rng()
    )
    @assert 0 < q < 1 "q must be between 0 and 1"
    @assert 0 <= tune_sigma <= 1 "tune_sigma must be between 0 and 1"
    @assert dims ∈ (1, 2) "dims must be 1 or 2"
    n_samples, n_features = size(data)
    # select the minimum values sample-wise (corresponding to the q-th quantile)
    min_vals = mapslices(data; dims) do x
        # edge case: all values in a slice are missing
        if all(ismissing, x)
            throw(
                error(
                    "All values in a slice are missing. This usually happens when " *
                        "there is a row or column with all missing values along the " *
                        "specified dimension. Please check your data."
                )
            )
        end
        quantile(skipmissing(x), q)
    end
    # estimate standard deviation using only features containing more than 50% non-NAs
    pNA_cols = sum(!ismissing, data; dims) ./ n_samples
    pNA_cols_mask = vec(pNA_cols .> 0.5)
    data_filtered = dims == 1 ? data[:, pNA_cols_mask] : data[pNA_cols_mask, :]
    sd_vals = mapslices(data_filtered; dims) do x
        std(skipmissing(x))
    end
    sd_temp = median(sd_vals) * tune_sigma
    # generate data from a normal distribution with the estimated parameters
    @views for i in 1:n_samples
        dist = Normal(min_vals[i], sd_temp)
        curr_sample = dims == 1 ? data[:, i] : data[i, :]
        curr_sample_imputed = trycopy(curr_sample)
        missing_idx = findall(ismissing, curr_sample)
        curr_sample_imputed[missing_idx] .= rand(rng, dist, n_features)[missing_idx]
        if dims == 1
            data[:, i] = curr_sample_imputed
        else
            data[i, :] = curr_sample_imputed
        end
    end
    return data
end


"""
    impute_median_cat(data::Matrix{<:Union{Missing, Real}})

Imputes missing elements based on a categorical imputation:
    - 0: Missing values
    - 1: Values below the median
    - 2: Values equal to or above the median
Returns a new matrix without modifying the original matrix.

# Arguments

- `data`: matrix of omics value, e.g., metabolomics matrix, where the rows are the samples
  and the columns are the features.
"""
function impute_median_cat(data::Matrix{<:Union{Missing, Real}})
    return impute_median_cat!(trycopy(data))
end

"""
    impute_median_cat!(data::Matrix{<:Union{Missing, Real}})

Imputes missing elements based on a categorical imputation:
    - 0: Missing values
    - 1: Values below the median
    - 2: Values equal to or above the median
Writes the result back to the original matrix.

# Arguments

- `data`: matrix of omics value, e.g., metabolomics matrix, where the rows are the samples
  and the columns are the features.
"""
function impute_median_cat!(data::Matrix{<:Union{Missing, Real}})
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
    imputeKNN(
        data::AbstractMatrix{Union{Missing, Float64}},
        k::Int = 1;
        threshold::Float64 = 0.5,
        dims::Union{Nothing, Int} = nothing,
        distance::NearestNeighbors.MinkowskiMetric = Euclidean()
    )

Replaces missing elements based on k-nearest neighbors (KNN). Returns a new matrix
without modifying the original matrix. This method is almost an exact copy of the KNN
imputation method from [Impute.jl](https://github.com/invenia/Impute.jl).

# Arguments

- `data`: matrix of omics value, e.g., metabolomics matrix, where the rows are the samples
  and the columns are the features.
- `k`: number of nearest neighbors to use for imputation.
- `threshold`: threshold for the number of missing neighbors above which the imputation is
  skipped.
- `dims`: dimension along which the statistic is calculated.
- `distance`: distance metric to use for the nearest neighbors search, taken from
  Distances.jl. Default is `Euclidean()`. This can only be one of the
  Minkowski metrics i.e. Euclidean, Cityblock, Minkowski and Chebyshev.
"""
function imputeKNN(
        data::AbstractMatrix{<:Union{Missing, Real}},
        k::Int = 1;
        threshold::Float64 = 0.5,
        dims::Union{Nothing, Int} = nothing,
        distance::NearestNeighbors.MinkowskiMetric = Euclidean()
    )
    # check arguments
    k < 1 && throw(ArgumentError("The number of nearset neighbors should be greater than 0"))
    !(0 < threshold < 1) &&
        throw(ArgumentError("Missing neighbors threshold should be within 0 to 1"))

    promoted = convert(Matrix{Union{Missing, Float64}}, data)
    return imputeKNN!(trycopy(promoted), k + 1, threshold, dims, distance)
end

"""
    imputeKNN!(
        data::AbstractMatrix{Union{Missing, Float64}},
        k::Int, threshold::Float64, dims::Union{Nothing, Int},
        distance::NearestNeighbors.MinkowskiMetric
    )

Replaces missing elements based on k-nearest neighbors (KNN) imputation. Writes the result
back to the original matrix. This method is almost an exact copy of the KNN imputation
method from [Impute.jl](https://github.com/invenia/Impute.jl).

# Arguments

- `data`: matrix of omics value, e.g., metabolomics matrix, where the rows are the samples
  and the columns are the features.
- `k`: number of nearest neighbors to use for imputation.
- `threshold`: threshold for the number of missing neighbors above which the imputation is
  skipped.
- `dims`: dimension along which the statistic is calculated.
- `distance`: distance metric to use for the nearest neighbors search, taken from
  Distances.jl. Default is `Euclidean()`. This can only be one of the
  Minkowski metrics i.e. Euclidean, Cityblock, Minkowski and Chebyshev.
"""
function imputeKNN!(
        data::AbstractMatrix{Union{Missing, Float64}},
        k::Int, threshold::Float64, dims::Union{Nothing, Int},
        distance::NearestNeighbors.MinkowskiMetric
    )
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

"""
    impute_QRILC(
        data::Matrix{<:Union{Missing, Real}};
        tune_sigma::Float64 = 1.0,
        eps::Float64 = 0.005
    )

Returns imputated matrix based on the "Quantile regression Imputation for left-censored
data" (QRILC) method. The function is based on the function `impute.QRILC` from the
`imputeLCMD` R package, with one difference: the default value of `eps` is set to 0.005
instead of 0.001.

# Arguments

- `data`: matrix of omics value, e.g., metabolomics matrix, where the rows are the samples
  and the columns are the features.
- `tune_sigma`: coefficient that controls the SD of the MNAR distribution:
                - 1 if the complete data distribution is supposed to be gaussian.
                - 0 < tune_sigma < 1 if the complete data distribution is supposed to be
                  left-censored.
  Default is 1.0.
- `eps`: small value added to the quantile for stability.
- `rng`: random number generator. Default is `Random.default_rng()`.
"""
function impute_QRILC(
        data::Matrix{<:Union{Missing, Real}};
        tune_sigma = 1.0, eps = 0.005,
        rng::AbstractRNG = Random.default_rng()
    )
    promoted = convert(Matrix{Union{Missing, Float64}}, data)
    return impute_QRILC!(trycopy(promoted); tune_sigma, eps, rng)
end

"""
    impute_QRILC!(
        data::Matrix{Union{Missing, Float64}};
        tune_sigma::Float64 = 1.0, eps::Float64 = 0.005,
        rng::AbstractRNG = Random.default_rng()
    )

Imputes missing elements based on the "Quantile regression Imputation for left-censored
data" (QRILC) method. Writes the result back to the original matrix. The function is based on
the function `impute.QRILC` from the `imputeLCMD` R package, with one difference: the
default value of `eps` is set to 0.005 instead of 0.001.

# Arguments

- `data`: matrix of omics value, e.g., metabolomics matrix, where the rows are the samples
  and the columns are the features.
- `tune_sigma`: coefficient that controls the SD of the MNAR distribution:
                - 1 if the complete data distribution is supposed to be gaussian.
                - 0 < tune_sigma < 1 if the complete data distribution is supposed to be
                  left-censored.
- `eps`: small value added to the quantile for stability.
- `rng`: random number generator. Default is `Random.default_rng()`.
"""
# TODO: elaborate on eps and why it is set to 0.005
function impute_QRILC!(
        data::Matrix{<:Union{Missing, Float64}};
        tune_sigma = 1.0, eps = 0.005,
        rng::AbstractRNG = Random.default_rng()
    )
    # Get dimensions of the data
    n_samples, n_features = size(data)
    @views for i in 1:n_samples
        curr_sample = data[:, i]
        # Calculate the percentage of missing values
        pNAs = count(ismissing, curr_sample) / length(curr_sample)
        # Estimate the mean and standard deviation of the original
        # distribution using quantile regression
        upper_q = 0.99
        q_normal = map(Base.Fix1(quantile, Normal(0, 1)), LinRange(pNAs + eps, upper_q + eps, 100))
        q_curr_sample = quantile(skipmissing(curr_sample), LinRange(eps, upper_q + eps, 100))
        temp_QR = lm(hcat(ones(length(q_normal), 1), reshape(q_normal, :, 1)), q_curr_sample)
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
        curr_sample_imputed[missing_idx] .= rand(rng, truncated_dist, n_features)[missing_idx]
        data[:, i] = curr_sample_imputed
    end
    return data
end

### TODO: add docstrings for the SVD imputation methods
function imputeSVD(
        data::AbstractMatrix{<:Union{Missing, Real}};
        rank::Union{Nothing, Int} = nothing,
        tol::Float64 = 1.0e-10,
        maxiter::Int = 100,
        limits::Union{Tuple{Float64, Float64}, Nothing} = nothing,
        dims::Union{Nothing, Int} = nothing,
        verbose::Bool = true
    )
    promoted = convert(Matrix{Union{Missing, Float64}}, data)
    return imputeSVD!(trycopy(promoted); rank, tol, maxiter, limits, dims, verbose)
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
    verbose && @debug(
        "Before", Diff = sum(mdata - mX), MAE = err, convergence = C,
        normsq = sum(abs2, mdata), mX[1]
    )

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
        verbose && @debug(
            "Iteration", i, Diff = sum(mdata - mX), MAE = err,
            MSS = sum(abs2, mdata), convergence = C
        )
        # Update missing values
        data[mmask] .= X[mmask]
        isfinite(C) && C < tol && break
    end

    return data
end

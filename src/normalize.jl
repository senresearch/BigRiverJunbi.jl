"""
    intnorm(mat::Matrix{<:Real}; dims::Int64 = 2, lambda::Real = 1)

Total Area Normalization for each row or column. By default, it normalizes each row.
This requires that the matrix has all positive values.

# Arguments
- `mat`: The matrix to normalize.
- `dims`: The dimension to normalize across. Default is 2.
- `lambda`: The lambda parameter for the normalization. Default is 1.

# Example

```jldoctest
julia> mat = [0.5 1 2 3 3.5;
              7 3 5 1.5 4.5;
              8 2 7 6 9]
3×5 Matrix{Float64}:
 0.5  1.0  2.0  3.0  3.5
 7.0  3.0  5.0  1.5  4.5
 8.0  2.0  7.0  6.0  9.0

julia> BigRiverJunbi.intnorm(mat)
3×5 Matrix{Float64}:
 0.05      0.1       0.2       0.3        0.35
 0.333333  0.142857  0.238095  0.0714286  0.214286
 0.25      0.0625    0.21875   0.1875     0.28125
```
"""
function intnorm(mat::Matrix{<:Real}; dims::Int64 = 2, lambda::Real = 1)
    # if matrix has any negative values, throw an error
    @assert all(mat .>= 0) "Matrix has negative values. Please remove negative values" *
        " before normalizing."
    return mat ./ (sum(mat; dims = dims) ./ lambda)
end

"""
    pqnorm(mat::Matrix{<:Real}; lambda::Real = 1)

Performs a probabilistic quotient normalization (PQN) for sample intensities.
This assumes that the matrix is organized as samples x features and requires that the
matrix have all positive values.

# Arguments
- `mat`: The matrix to normalize.

# Example

```jldoctest
julia> mat = [0.5 1 2 3 3.5;
              7 3 5 1.5 4.5;
              8 2 7 6 9]
3×5 Matrix{Float64}:
 0.5  1.0  2.0  3.0  3.5
 7.0  3.0  5.0  1.5  4.5
 8.0  2.0  7.0  6.0  9.0

julia> BigRiverJunbi.pqnorm(mat)
3×5 Matrix{Float64}:
 0.05     0.1      0.2      0.3       0.35
 0.30625  0.13125  0.21875  0.065625  0.196875
 0.25     0.0625   0.21875  0.1875    0.28125
```
"""
function pqnorm(mat::Matrix{<:Real}; lambda::Real = 1)
    # if matrix has any negative values, throw an error
    @assert all(mat .>= 0) "Matrix has negative values. Please remove negative values" *
        " before normalizing."
    # Integral normalization
    mat = intnorm(mat; dims = 2, lambda)
    # Calculate the reference spectrum (default: median) of all samples
    ref_spec = median(mat; dims = 1)
    # Calculate the quotients of all variables of interest of the test spectrum
    # with those of the reference spectrum.
    quotients = mat ./ ref_spec
    # Calculate the median of these quotients.
    med_quotients = median(quotients; dims = 2)
    # Divide all variables of the test spectrum by this median
    mat_norm = mat ./ med_quotients
    return mat_norm
end

"""
    quantilenorm(data::Matrix{<:Real})

Performs quantile normalization for sample intensities. This assumes
that the matrix is organized as samples x features.

# Arguments
- `data`: The matrix to normalize.

# Example

```jldoctest
julia> mat = [0.5 1 2 3 3.5;
              7 3 5 1.5 4.5;
              8 2 7 6 9]
3×5 Matrix{Float64}:
 0.5  1.0  2.0  3.0  3.5
 7.0  3.0  5.0  1.5  4.5
 8.0  2.0  7.0  6.0  9.0

julia> BigRiverJunbi.quantilenorm(mat)
3×5 Matrix{Float64}:
 1.7  1.7  1.7  4.3  1.7
 4.3  6.6  4.3  1.7  4.3
 6.6  4.3  6.6  6.6  6.6
```
"""
function quantilenorm(data::Matrix{<:Real})
    # creates a matrix of ranks for each column
    ranks = reduce(hcat, StatsBase.competerank.(eachcol(data)))
    # sort each column by the ranks
    data_sorted = sort(data; dims = 1)
    # find the mean of each row and rank them
    data_mean = mean(data_sorted; dims = 2)
    data_sorted_idxs = sortperm(data_mean; dims = 1)
    data_mean_ranked = Dict(zip(data_sorted_idxs, data_mean))
    results = mapslices(ranks, dims = 1) do x
        get.(Ref(data_mean_ranked), x, missing)
    end
    return results
end

"""
    huberize(
        mat::Matrix{<:Real}; alpha::Real = 1,
        error_on_zero_mad::Bool = true
    )

Performs Huberization for sample intensities.

# Arguments
- `mat`: The matrix to normalize.
- `alpha`: The alpha parameter for Huberization. Default is 1.
- `error_on_zero_mad`: Whether to throw an error if the MAD is zero. Default is `true`.

!!! warning
    If you set `error_on_zero_mad` to `false`, this function will return a result with NaN
    values if the MAD is zero. This can be useful if you are expecting this behavior and
    want to handle it yourself, but should be used with caution.

# Example

```jldoctest
julia> mat = [0.5 1 2 3 3.5;
              7 3 5 1.5 4.5;
              8 2 7 6 9]
3×5 Matrix{Float64}:
 0.5  1.0  2.0  3.0  3.5
 7.0  3.0  5.0  1.5  4.5
 8.0  2.0  7.0  6.0  9.0

julia> BigRiverJunbi.huberize(mat)
3×5 Matrix{Float64}:
 2.86772  1.0  2.0002  3.0      3.5
 7.0      3.0  5.0     1.5      4.5
 8.0      2.0  7.0     5.89787  7.83846
```
"""
function huberize(mat::Matrix{<:Real}; alpha::Real = 1, error_on_zero_mad::Bool = true)
    # check if the MAD is zero for each column and throw an error if it is
    # disable this check if error_on_zero_mad is false
    error_on_zero_mad && check_mad(mat; dims = 2)
    # apply Huberization to each column
    return mapslices(mat, dims = 1) do x
        huberize(x; alpha, error_on_zero_mad)
    end
end

"""
    huberize(
        x::Vector{<:Real}; alpha::Real = 1,
        error_on_zero_mad::Bool = true
    )

Performs Huberization for a single vector.

# Arguments
- `x`: The vector to Huberize.
- `alpha`: The alpha parameter for the Huberization. Default is 1.0.
- `error_on_zero_mad`: Whether to throw an error if the MAD is zero. Default is `true`.

!!! warning
    If you set `error_on_zero_mad` to `false`, this function will return a result with NaN
    values if the MAD is zero. This can be useful if you are expecting this behavior and
    want to handle it yourself, but should be used with caution.
"""
function huberize(x::Vector{<:Real}; alpha::Real = 1, error_on_zero_mad::Bool = true)
    error_on_zero_mad && check_mad(x)
    med = median(x)
    s = mad(x; center = med, normalize = true)
    z = (x .- med) ./ s
    l = huberloss.(z; alpha)
    x = sign.(z) .* sqrt.(2l)
    return med .+ s .* x
end

"""
    huberloss(x::Real; alpha::Real = 1)

Computes the Huber loss for a given value. This is defined as:

```math
L(x) = \\begin{cases}
    \\frac{1}{2}x^2 & \\text{if } |x| \\leq \\alpha \\\\
    \\alpha (|x| - \\frac{\\alpha^2}{2}) & \\text{if } |x| > \\alpha
\\end{cases}
```

# Arguments
- `x`: The value to compute the Huber loss for.
- `alpha`: The alpha parameter for the Huber loss. Default is 1.
"""
function huberloss(x::Real; alpha::Real = 1)
    @assert alpha > 0 "Huber crossover parameter alpha must be positive."
    d = abs(x)
    if d <= alpha
        return d^2 / 2
    else
        return alpha * (d - alpha^2 / 2)
    end
end

"""
    standardize(mat::Matrix{<:Real}; center::Bool = true)

Standardize a matrix i.e. scale to unit variance, with the option of centering or not.

# Arguments
- `mat`: The matrix to standardize.
- `center`: Whether to center the data. Default is `true`.
"""
function standardize(mat::Matrix{<:Real}; center::Bool = true)
    dt = fit(ZScoreTransform, mat; dims = 1, center)
    return StatsBase.transform(dt, mat)
end

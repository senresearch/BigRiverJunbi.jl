"""
    log_tx(mat::Matrix{<:Real}; base::Real = exp(1), log_offset::Real = 0.0)

Compute the logarithm of `mat`, adding `log_offset` to all values (for instance, to avoid
`log(0)`). Returns a new matrix without modifying the original.

# Arguments
- `mat`: The matrix to transform.
- `base`: The base of the logarithm. Default is `exp(1)`.
- `log_offset`: The constant to add to all values. Default is `0.0`.

# Example

```jldoctest
julia> mat = [0.5 1 2 3 3.5;
             7 3 5 0 3.5;
             8 2 5 6 0]
3×5 Matrix{Float64}:
 0.5  1.0  2.0  3.0  3.5
 7.0  3.0  5.0  0.0  3.5
 8.0  2.0  5.0  6.0  0.0

julia> BigRiverJunbi.log_tx(mat; log_offset = 1)
3×5 Matrix{Float64}:
 0.584963  1.0      1.58496  2.0      2.16993
 3.0       2.0      2.58496  0.0      2.16993
 3.16993   1.58496  2.58496  2.80735  0.0

 ---

  log_tx(data::AbstractMatrix{<:Union{Missing, Real}};
        base::Real = exp(1),
        log_offset::Real = 0.0)

Apply a logarithmic transform to the non-missing entries of `data`
after shifting by `log_offset`. Returns the log transformed `data`.

# Arguments
- `data`: Matrix-like container mutated in place.
- `base`: The base of the logarithm. Default is `exp(1)`.
- `log_offset`: Constant added prior to taking the logarithm. Must ensure `x + log_offset > 0`.

# Throws
- `ArgumentError` if a non-missing entry violates `x + log_offset > 0`.


```
"""
function log_tx(
        data::AbstractMatrix{<:Real};
        base::Real = 2,
        log_offset::Real = 0.0
    )
    result = copy(data)
    return log_tx!(result; base = base, log_offset = log_offset)
end

function log_tx(
        data::AbstractMatrix{<:Union{Missing, Real}};
        base::Real = 2,
        log_offset::Real = 0.0
    )
    result = copy(data)
    return log_tx!(result; base = base, log_offset = log_offset)
end


"""
log_tx(mat::Matrix{<:Real}; base::Real = exp(1), log_offset::Real = 0.0)

Apply an in-place logarithmic transform after shifting by `log_offset`.

# Arguments
- `mat`: The matrix to transform.
- `base`: The base of the logarithm. Default is `exp(1)`.
- `log_offset`: The constant to add to all values. Default is `0.0`.

# Example

```jldoctest
julia> mat = [0.5 1 2 3 3.5;
             7 3 5 0 3.5;
             8 2 5 6 0]
3×5 Matrix{Float64}:
 0.5  1.0  2.0  3.0  3.5
 7.0  3.0  5.0  0.0  3.5
 8.0  2.0  5.0  6.0  0.0

julia> BigRiverJunbi.log_tx(mat; log_offset = 1)
3×5 Matrix{Float64}:
 0.584963  1.0      1.58496  2.0      2.16993
 3.0       2.0      2.58496  0.0      2.16993
 3.16993   1.58496  2.58496  2.80735  0.0

julia> mat
3×5 Matrix{Float64}:
 0.584963  1.0      1.58496  2.0      2.16993
 3.0       2.0      2.58496  0.0      2.16993
 3.16993   1.58496  2.58496  2.80735  0.0

 ---

  log_tx!(data::AbstractMatrix{<:Union{Missing, Real}};
        base::Real = exp(1),
        log_offset::Real = 0.0)

Apply an in-place logarithmic transform after shifting by `log_offset`. For matrices that may
contain `missing`, only the non-missing entries are transformed.

# Arguments
- `data`: Matrix-like container mutated in place.
- `base`: The base of the logarithm. Default is `exp(1)`.
- `log_offset`: Constant added prior to taking the logarithm. Must ensure `x + log_offset > 0`.

# Throws
- `ArgumentError` if a non-missing entry violates `x + log_offset > 0`.

Returns the mutated `data`.
"""
function log_tx!(
        data::Matrix{<:Real};
        base::Real = exp(1),
        log_offset::Real = 0.0
    )
    b = float(base)
    if !(b > 0) || b == 1
        throw(ArgumentError("`base` must be > 0 and ≠ 1; got $base"))
    end

    @inbounds for j in axes(data, 2), i in axes(data, 1)
        s = data[i, j] + log_offset
        if !(s > 0)
            throw(
                ArgumentError(
                    "log_tx! requires x + log_offset > 0; got $(data[i, j]) at ($(i),$(j)). Increase log_offset."
                )
            )
        end
        data[i, j] = log(b, s)
    end

    return data
end


function log_tx!(
        data::AbstractMatrix{<:Union{Missing, Real}};
        base::Real = exp(1),
        log_offset::Real = 0.0
    )

    # validate base
    b = float(base)
    if !(b > 0) || b == 1
        throw(ArgumentError("`base` must be > 0 and ≠ 1; got $base"))
    end

    @inbounds for j in axes(data, 2), i in axes(data, 1)
        v = data[i, j]
        if !ismissing(v)
            s = v + log_offset
            if !(s > 0)
                throw(
                    ArgumentError(
                        "log_tx! requires x + log_offset > 0; got $(v) at ($(i),$(j)). Increase log_offset."
                    )
                )
            end
            data[i, j] = log(base, s)
        end
    end

    return data
end


"""
    log_tx_inv(mat::Matrix{<:Real}; base::Real = exp(1), log_offset::Real = 0.0)

Undo [`log_tx`](@ref) by exponentiating each entry in `mat` (with respect to `base`) and
subtracting `log_offset`. Returns a new matrix, leaving `mat` unchanged.
"""
function log_tx_inv(
        mat::Matrix{<:Real};
        base::Real = exp(1),
        log_offset::Real = 0.0
    )
    result = copy(mat)
    return log_tx_inv!(result; base = base, log_offset = log_offset)
end

function log_tx_inv(
        data::AbstractMatrix{<:Union{Missing, Real}};
        base::Real = exp(1),
        log_offset::Real = 0.0
    )
    result = copy(data)
    return log_tx_inv!(result; base = base, log_offset = log_offset)
end

"""
    log_tx_inv!(data::Matrix{<:Real}; base::Real = exp(1), log_offset::Real = 0.0)

In-place inverse of [`log_tx!`](@ref) for matrices without `missing` values.
"""
function log_tx_inv!(
        data::Matrix{<:Real};
        base::Real = exp(1),
        log_offset::Real = 0.0
    )
    b = float(base)
    if !(b > 0) || b == 1
        throw(ArgumentError("`base` must be > 0 and ≠ 1; got $base"))
    end
    log_b = log(b)

    @inbounds for j in axes(data, 2), i in axes(data, 1)
        data[i, j] = exp(log_b * data[i, j]) - log_offset
    end

    return data
end

"""
    log_tx_inv!(data::AbstractMatrix{<:Union{Missing, Real}}; base::Real = exp(1), log_offset::Real = 0.0)

In-place inverse of [`log_tx!`](@ref) that skips entries equal to `missing`.
"""
function log_tx_inv!(
        data::AbstractMatrix{<:Union{Missing, Real}};
        base::Real = exp(1),
        log_offset::Real = 0.0
    )
    b = float(base)
    if !(b > 0) || b == 1
        throw(ArgumentError("`base` must be > 0 and ≠ 1; got $base"))
    end
    log_b = log(b)

    @inbounds for j in axes(data, 2), i in axes(data, 1)
        v = data[i, j]
        if !ismissing(v)
            data[i, j] = exp(log_b * v) - log_offset
        end
    end

    return data
end


"""
    meancenter_tx(mat::Matrix{Float64}, dims::Int64 = 1)

Mean center a matrix across the specified dimension. This requires that the matrix has
all positive values.

# Arguments
- `mat`: The matrix to transform.
- `dims`: The dimension to mean center across. Default is 1.

# Example

```jldoctest
julia> mat = [0.5 1 2 3 3.5;
             7 3 5 0 3.5;
             8 2 5 6 0]
3×5 Matrix{Float64}:
 0.5  1.0  2.0  3.0  3.5
 7.0  3.0  5.0  0.0  3.5
 8.0  2.0  5.0  6.0  0.0

julia> BigRiverJunbi.meancenter_tx(mat)
3×5 Matrix{Float64}:
 -4.66667  -1.0  -2.0   0.0   1.16667
  1.83333   1.0   1.0  -3.0   1.16667
  2.83333   0.0   1.0   3.0  -2.33333
```
"""
meancenter_tx(mat::Matrix{<:Real}, dims::Int64 = 1) = mat .- mean(mat; dims)

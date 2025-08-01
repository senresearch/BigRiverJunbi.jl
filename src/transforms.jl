"""
    log_tx(mat::Matrix{<:Real}; base::Real = 2, constant::Real = 0)

Computes logarithm on a matrix, adding a constant to all values (for instance, to avoid log(0)).
Default base is 2, default constant is 0.

# Arguments
- `mat`: The matrix to transform.
- `base`: The base of the logarithm. Default is 2.
- `constant`: The constant to add to all values. Default is 0.

# Example

```jldoctest
julia> mat = [0.5 1 2 3 3.5;
             7 3 5 0 3.5;
             8 2 5 6 0]
3×5 Matrix{Float64}:
 0.5  1.0  2.0  3.0  3.5
 7.0  3.0  5.0  0.0  3.5
 8.0  2.0  5.0  6.0  0.0

julia> BigRiverJunbi.log_tx(mat; constant = 1)
3×5 Matrix{Float64}:
 0.584963  1.0      1.58496  2.0      2.16993
 3.0       2.0      2.58496  0.0      2.16993
 3.16993   1.58496  2.58496  2.80735  0.0
```
"""
function log_tx(mat::Matrix{<:Real}; base::Real = 2, constant::Real = 0)
    mat = mat .+ constant
    @assert all(mat .> 0) "Matrix has non-positive values even after adding constant. Please" *
        " remove such values before transforming."
    return log.(base, mat)
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

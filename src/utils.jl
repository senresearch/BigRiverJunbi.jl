# Check if the MAD is zero for each column of the matrix. If it is, then errors and
# displays the list of columns with zero MAD. Can also be used to check each row of a matrix
# by setting `dims` to 1.
function check_mad(mat::Matrix{<:Real}; dims::Int = 2)
    @assert dims in [1, 2] "dims must be 1 or 2"
    error_cols = String[]
    for i in axes(mat, dims)
        try
            dims == 2 ? check_mad(mat[:, i]) : check_mad(mat[i, :])
        catch
            push!(error_cols, string(i))
        end
    end
    return if length(error_cols) > 0
        throw(
            ErrorException(
                "The MAD (median absolute deviation) of the following " *
                    "slices along dimension $dims: $error_cols is zero, which " *
                    "implies that some of the data is very close to the median. " *
                    "the data is very close to the median. Please check your " *
                    "data."
            )
        )
    end
end

# Checks if the MAD (median absolute deviation) is zero for a vector. If it is, then errors.
function check_mad(x::Vector{<:Real})
    s = mad(x; normalize = true)
    return if s == 0
        throw(
            ErrorException(
                "The MAD (median absolute deviation) of this vector is zero, " *
                    "which implies that some of the data is very close to the " *
                    "median. Please check your data."
            )
        )
    end
end

# utility function to copy data – copy is more performant than deepcopy
# but use deepcopy if copy fails
function trycopy(data)
    return try
        copy(data)
    catch
        deepcopy(data)
    end
end

### Specifically for DataFramesExt
# This is a hack to allow new functions to be defined in the extension without
# having to define them in the main module.
for fn_name in [:missing_summary, :missing_percentages]
    @eval function $fn_name(df)
        ext = Base.get_extension(@__MODULE__, :DataFramesExt)
        if !isnothing(ext)
            return ext.$fn_name(df)
        else
            throw(
                ErrorException(
                    "Using this function requires `DataFrames` to be loaded, " *
                        "since it is in an extension. Please load DataFrames " *
                        "with `using DataFrames` and then use the function."
                )
            )
        end
    end
end

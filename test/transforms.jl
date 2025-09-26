@testitem "log_tx basic functionality" begin
    using Test
    using BigRiverJunbi
    using Statistics

    # Test basic functionality from doctest example
    mat = [
        0.5 1 2 3 3.5;
        7 3 5 0 3.5;
        8 2 5 6 0
    ]

    # Test with constant to avoid log(0)
    result = BigRiverJunbi.log_tx(mat; log_offset = 1)
    expected = [
        0.584963 1.0 1.58496 2.0 2.16993;
        3.0 2.0 2.58496 0.0 2.16993;
        3.16993 1.58496 2.58496 2.80735 0.0
    ]
    @test result ≈ expected rtol = 1.0e-4

    # Test with different base
    mat_positive = [1.0 2.0 4.0; 8.0 16.0 32.0]
    result_base2 = BigRiverJunbi.log_tx(mat_positive; base = 2)
    expected_base2 = [0.0 1.0 2.0; 3.0 4.0 5.0]
    @test result_base2 ≈ expected_base2

    # Test with base 10
    result_base10 = BigRiverJunbi.log_tx(mat_positive; base = 10)
    expected_base10 = log.(10, mat_positive)
    @test result_base10 ≈ expected_base10

    # Test with base e (natural log)
    result_base_e = BigRiverJunbi.log_tx(mat_positive; base = ℯ)
    expected_base_e = log.(mat_positive)
    @test result_base_e ≈ expected_base_e

    # Test with no constant (default)
    mat_no_zeros = [1.0 2.0; 3.0 4.0]
    result_no_constant = BigRiverJunbi.log_tx(mat_no_zeros)
    expected_no_constant = log.(2, mat_no_zeros)
    @test result_no_constant ≈ expected_no_constant
end

@testitem "log_tx error cases" begin
    using Test
    using BigRiverJunbi

    # Test error with negative values even after adding constant
    mat_negative = [-2.0 1.0; 3.0 -1.0]
    @test_throws AssertionError BigRiverJunbi.log_tx(mat_negative; log_offset = 1)

    # Test error with zero values and no constant
    mat_with_zeros = [0.0 1.0; 2.0 3.0]
    @test_throws AssertionError BigRiverJunbi.log_tx(mat_with_zeros)

    # Test error with negative values and no constant
    mat_with_negative = [-1.0 1.0; 2.0 3.0]
    @test_throws AssertionError BigRiverJunbi.log_tx(mat_with_negative)

    # Test error when constant is insufficient
    mat_very_negative = [-5.0 1.0; 2.0 -3.0]
    @test_throws AssertionError BigRiverJunbi.log_tx(mat_very_negative; log_offset = 2)

    # Test behavior with invalid base
    mat_positive = [1.0 2.0; 3.0 4.0]
    @test_throws DomainError BigRiverJunbi.log_tx(mat_positive; base = -1)

    # Test that base = 0 produces -0.0 values
    result_base0 = BigRiverJunbi.log_tx(mat_positive; base = 0)
    @test all(result_base0 .== -0.0)

    # Test that base = 1 produces NaN and Inf values
    result_base1 = BigRiverJunbi.log_tx(mat_positive; base = 1)
    @test isnan(result_base1[1, 1])  # log₁(1) = NaN
    @test all(isinf.(result_base1[1, 2:end]))  # log₁(x) for x > 1 = Inf
end

@testitem "log_tx edge cases" begin
    using Test
    using BigRiverJunbi

    # Test with single element matrix
    mat_single = reshape([2.0], 1, 1)
    result_single = BigRiverJunbi.log_tx(mat_single)
    @test result_single ≈ reshape([1.0], 1, 1)  # log₂(2) = 1

    # Test with large constant
    mat_small = [0.1 0.2; 0.3 0.4]
    result_large_constant = BigRiverJunbi.log_tx(mat_small; log_offset = 100)
    expected_large_constant = log.(2, mat_small .+ 100)
    @test result_large_constant ≈ expected_large_constant

    # Test with integer matrix
    mat_int = [1 2; 4 8]
    result_int = BigRiverJunbi.log_tx(mat_int)
    expected_int = [0.0 1.0; 2.0 3.0]
    @test result_int ≈ expected_int

    # Test type preservation behavior
    mat_float32 = Float32[1.0 2.0; 4.0 8.0]
    result_float32 = BigRiverJunbi.log_tx(mat_float32)
    @test result_float32 isa Matrix{Float32}

    # Test with very small positive values
    mat_tiny = [1.0e-10 1.0e-5; 1.0e-3 1.0]
    result_tiny = BigRiverJunbi.log_tx(mat_tiny; log_offset = 0)
    # Should not throw error since all values are positive
    @test all(isfinite.(result_tiny))
end

@testitem "log_tx with missing values" begin
    using Test
    using BigRiverJunbi

    mat = Union{Missing, Float64}[1.0 missing; 3.0 5.0]
    mat_copy = copy(mat)
    result = BigRiverJunbi.log_tx(mat; base = ℯ, log_offset = 1.0)

    @test isequal(mat, mat_copy)  # non-mutating
    @test result !== mat  # returns new matrix
    @test result[1, 1] ≈ log(ℯ, 2.0)
    @test result[2, 1] ≈ log(ℯ, 4.0)
    @test result[2, 2] ≈ log(ℯ, 6.0)
    @test ismissing(result[1, 2])

    mat_bad = Union{Missing, Float64}[missing -0.5; 0.0 1.0]
    @test_throws ArgumentError BigRiverJunbi.log_tx(mat_bad; log_offset = 0.0)
end

@testitem "log_tx! in-place real matrices" begin
    using Test
    using BigRiverJunbi

    mat = [1.0 2.0; 4.0 8.0]
    BigRiverJunbi.log_tx!(mat; base = 2, log_offset = 0.0)
    @test mat ≈ [0.0 1.0; 2.0 3.0]

    mat = [0.1 1.0; 2.0 3.0]
    @test_throws ArgumentError BigRiverJunbi.log_tx!(mat; log_offset = 0.0)

    mat = [0.1 1.0; 2.0 3.0]
    mat_copy = copy(mat)
    BigRiverJunbi.log_tx!(mat; log_offset = 1.0)
    @test mat ≈ log.(ℯ, mat_copy .+ 1.0)
end

@testitem "log_tx_inv round trip" begin
    using Test
    using BigRiverJunbi

    mat = [0.25 1.5; 4.0 8.0]
    transformed = BigRiverJunbi.log_tx(mat; base = 2, log_offset = 0.75)
    restored = BigRiverJunbi.log_tx_inv(transformed; base = 2, log_offset = 0.75)
    @test restored ≈ mat

    mat_miss = Union{Missing, Float64}[missing 1.0; 2.5 missing]
    transformed_miss = BigRiverJunbi.log_tx(mat_miss; log_offset = 0.5)
    restored_miss = BigRiverJunbi.log_tx_inv(transformed_miss; log_offset = 0.5)
    @test isequal(restored_miss, mat_miss)

    # Mutating inverse
    mat_roundtrip = copy(mat)
    BigRiverJunbi.log_tx!(mat_roundtrip; base = 2, log_offset = 0.75)
    BigRiverJunbi.log_tx_inv!(mat_roundtrip; base = 2, log_offset = 0.75)
    @test mat_roundtrip ≈ mat
end

@testitem "meancenter_tx basic functionality" begin
    using Test
    using BigRiverJunbi
    using Statistics

    # Test basic functionality from doctest example
    mat = [
        0.5 1 2 3 3.5;
        7 3 5 0 3.5;
        8 2 5 6 0
    ]

    result = BigRiverJunbi.meancenter_tx(mat)
    expected = [
        -4.66667 -1.0 -2.0 0.0 1.16667;
        1.83333 1.0 1.0 -3.0 1.16667;
        2.83333 0.0 1.0 3.0 -2.33333
    ]
    @test result ≈ expected rtol = 1.0e-4

    # Test that column means are approximately zero after centering (dims=1)
    col_means = mean(result; dims = 1)
    @test all(abs.(col_means) .< 1.0e-10)

    # Test column-wise centering (dims=2)
    result_dims2 = BigRiverJunbi.meancenter_tx(mat, 2)
    expected_dims2 = mat .- mean(mat; dims = 2)
    @test result_dims2 ≈ expected_dims2

    # Test that row means are approximately zero after centering (dims=2)
    row_means = mean(result_dims2; dims = 2)
    @test all(abs.(row_means) .< 1.0e-10)

    # Verify manual calculation matches function
    manual_result = mat .- mean(mat; dims = 1)
    @test result ≈ manual_result
end

@testitem "meancenter_tx different dimensions" begin
    using Test
    using BigRiverJunbi
    using Statistics

    # Test with 2x2 matrix
    mat_small = [1.0 3.0; 5.0 7.0]

    # Test dims=1 (row-wise centering, subtract row means)
    result_dims1 = BigRiverJunbi.meancenter_tx(mat_small, 1)
    expected_dims1 = mat_small .- mean(mat_small; dims = 1)
    @test result_dims1 ≈ expected_dims1

    # Test dims=2 (column-wise centering, subtract column means)
    result_dims2 = BigRiverJunbi.meancenter_tx(mat_small, 2)
    expected_dims2 = mat_small .- mean(mat_small; dims = 2)
    @test result_dims2 ≈ expected_dims2

    # Test single row matrix
    mat_single_row = reshape([1.0, 2.0, 3.0, 4.0], 1, 4)
    result_single_row = BigRiverJunbi.meancenter_tx(mat_single_row)
    expected_single_row = mat_single_row .- mean(mat_single_row; dims = 1)
    @test result_single_row ≈ expected_single_row

    # Test single column matrix
    mat_single_col = reshape([1.0, 2.0, 3.0, 4.0], 4, 1)
    result_single_col = BigRiverJunbi.meancenter_tx(mat_single_col)
    expected_single_col = mat_single_col .- mean(mat_single_col; dims = 1)
    @test result_single_col ≈ expected_single_col
end

@testitem "meancenter_tx edge cases and type handling" begin
    using Test
    using BigRiverJunbi
    using Statistics

    # Test with single element matrix
    mat_single = reshape([5.0], 1, 1)
    result_single = BigRiverJunbi.meancenter_tx(mat_single)
    @test result_single ≈ reshape([0.0], 1, 1)

    # Test with integer matrix
    mat_int = [1 2 3; 4 5 6]
    result_int = BigRiverJunbi.meancenter_tx(mat_int)
    expected_int = mat_int .- mean(mat_int; dims = 1)
    @test result_int ≈ expected_int

    # Test with negative values
    mat_negative = [-1.0 -2.0; -3.0 -4.0]
    result_negative = BigRiverJunbi.meancenter_tx(mat_negative)
    expected_negative = mat_negative .- mean(mat_negative; dims = 1)
    @test result_negative ≈ expected_negative

    # Test with mixed positive and negative values
    mat_mixed = [-2.0 3.0 -1.0; 4.0 -5.0 6.0]
    result_mixed = BigRiverJunbi.meancenter_tx(mat_mixed)
    expected_mixed = mat_mixed .- mean(mat_mixed; dims = 1)
    @test result_mixed ≈ expected_mixed

    # Test with zero values
    mat_zeros = [0.0 1.0 0.0; 2.0 0.0 3.0]
    result_zeros = BigRiverJunbi.meancenter_tx(mat_zeros)
    expected_zeros = mat_zeros .- mean(mat_zeros; dims = 1)
    @test result_zeros ≈ expected_zeros

    # Test that centering preserves matrix dimensions
    original_size = size(mat_zeros)
    result_size = size(result_zeros)
    @test original_size == result_size
end

@testitem "meancenter_tx error cases" begin
    using Test
    using BigRiverJunbi
    using Statistics

    # Test with invalid dims parameter
    mat = [1.0 2.0; 3.0 4.0]

    # dims=0 should cause error (dims must be 1 or 2 for 2D matrix)
    @test_throws ErrorException BigRiverJunbi.meancenter_tx(mat, 0)

    # dims=3 works without error in Julia (returns same matrix)
    result_dims3 = BigRiverJunbi.meancenter_tx(mat, 3)
    @test result_dims3 ≈ mat .- mean(mat; dims = 3)

    # dims=-1 should cause error
    @test_throws ErrorException BigRiverJunbi.meancenter_tx(mat, -1)
end

@testitem "transforms property tests" begin
    using Test
    using BigRiverJunbi
    using Statistics

    # Property test: log_tx should be monotonic
    mat_ascending = [1.0 2.0 3.0; 4.0 5.0 6.0]
    result_log = BigRiverJunbi.log_tx(mat_ascending)

    # Check monotonicity within each row
    for i in 1:size(mat_ascending, 1)
        @test issorted(result_log[i, :])
    end

    # Property test: meancenter_tx should preserve column-wise differences
    mat_test = [1.0 4.0 7.0; 2.0 5.0 8.0]
    result_center = BigRiverJunbi.meancenter_tx(mat_test)

    # Column-wise differences between rows should be preserved
    original_diff = mat_test[2, 1] - mat_test[1, 1]
    centered_diff = result_center[2, 1] - result_center[1, 1]
    @test original_diff ≈ centered_diff

    # Property test: log_tx with constant=0 should equal natural log behavior
    mat_positive = [1.0 ℯ ℯ^2; ℯ^3 ℯ^4 ℯ^5]
    result_natural = BigRiverJunbi.log_tx(mat_positive; base = ℯ)
    expected_natural = [0.0 1.0 2.0; 3.0 4.0 5.0]
    @test result_natural ≈ expected_natural rtol = 1.0e-10

    # Property test: mean centering should not change variance patterns
    mat_variance_test = randn(5, 4)  # Random matrix
    original_vars = var(mat_variance_test; dims = 1)
    centered = BigRiverJunbi.meancenter_tx(mat_variance_test)
    centered_vars = var(centered; dims = 1)
    @test original_vars ≈ centered_vars rtol = 1.0e-10
end

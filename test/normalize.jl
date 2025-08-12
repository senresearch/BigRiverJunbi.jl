@testitem "intnorm" begin
    # Test basic functionality from doctest example
    mat = [
        0.5 1 2 3 3.5;
        7 3 5 1.5 4.5;
        8 2 7 6 9
    ]

    expected = [
        0.05 0.1 0.2 0.3 0.35;
        0.333333 0.142857 0.238095 0.0714286 0.214286;
        0.25 0.0625 0.21875 0.1875 0.28125
    ]

    result = BigRiverJunbi.intnorm(mat)
    @test result ≈ expected rtol = 1.0e-5

    # Test column-wise normalization (dims=1)
    result_col = BigRiverJunbi.intnorm(mat; dims = 1)
    expected_col = mat ./ sum(mat; dims = 1)
    @test result_col ≈ expected_col

    # Test with different lambda
    result_lambda = BigRiverJunbi.intnorm(mat; lambda = 2.0)
    expected_lambda = mat ./ (sum(mat; dims = 2) ./ 2.0)
    @test result_lambda ≈ expected_lambda

    # Test error with negative values
    mat_negative = [1 -2; 3 4]
    @test_throws AssertionError BigRiverJunbi.intnorm(mat_negative)

    # Test with zeros (should work)
    mat_zeros = [1 0 2; 0 3 1]
    result_zeros = BigRiverJunbi.intnorm(mat_zeros)
    @test all(sum(result_zeros; dims = 2) .≈ 1.0)
end

@testitem "pqnorm" begin
    # Test basic functionality from doctest example
    mat = [
        0.5 1 2 3 3.5;
        7 3 5 1.5 4.5;
        8 2 7 6 9
    ]

    expected = [
        0.05 0.1 0.2 0.3 0.35;
        0.30625 0.13125 0.21875 0.065625 0.196875;
        0.25 0.0625 0.21875 0.1875 0.28125
    ]

    result = BigRiverJunbi.pqnorm(mat)
    @test result ≈ expected rtol = 1.0e-5

    # Test error with negative values
    mat_negative = [1 -2; 3 4]
    @test_throws AssertionError BigRiverJunbi.pqnorm(mat_negative)

    # Test with single row
    mat_single = reshape([1.0, 2.0, 3.0, 4.0, 5.0], 1, 5)
    result_single = BigRiverJunbi.pqnorm(mat_single)
    @test size(result_single) == (1, 5)
    @test all(result_single .> 0)
end

@testitem "quantilenorm" begin
    # Test basic functionality from doctest example
    mat = [
        0.5 1 2 3 3.5;
        7 3 5 1.5 4.5;
        8 2 7 6 9
    ]

    expected = [
        1.7 1.7 1.7 4.3 1.7;
        4.3 6.6 4.3 1.7 4.3;
        6.6 4.3 6.6 6.6 6.6
    ]

    result = BigRiverJunbi.quantilenorm(mat)
    @test result ≈ expected rtol = 1.0e-10

    # Test with identical columns
    mat_identical = [1.0 1.0; 2.0 2.0; 3.0 3.0]
    result_identical = BigRiverJunbi.quantilenorm(mat_identical)
    @test result_identical[:, 1] ≈ result_identical[:, 2]

    # Test with single column
    mat_single_col = reshape([1.0, 3.0, 2.0], 3, 1)
    result_single_col = BigRiverJunbi.quantilenorm(mat_single_col)
    @test size(result_single_col) == (3, 1)

    # Test that ranks are preserved within each feature (column)
    @views for j in axes(mat, 2)
        original_ranks = sortperm(mat[:, j])
        result_ranks = sortperm(result[:, j])
        @test original_ranks == result_ranks
    end
end

@testitem "huberize matrix" begin
    # Test normal case without MAD issues
    mat = [
        0.5 1 2 3 3.5;
        7 3 5 1.5 4.5;
        8 2 7 6 9
    ]

    expected = [
        2.86772 1.0 2.0002 3.0 3.5;
        7.0 3.0 5.0 1.5 4.5;
        8.0 2.0 7.0 5.89787 7.83846
    ]

    result = BigRiverJunbi.huberize(mat)
    @test result ≈ expected rtol = 1.0e-4

    # Test with different alpha
    result_alpha = BigRiverJunbi.huberize(mat; alpha = 0.5)
    @test size(result_alpha) == size(mat)
    @test result_alpha != result  # Should be different with different alpha

    # Test with error_on_zero_mad = false
    mat_zero_mad = [1.0 2.0 2.0; 2.0 2.0 2.0; 3.0 2.0 2.0]
    result_no_error = BigRiverJunbi.huberize(mat_zero_mad; error_on_zero_mad = false)
    @test size(result_no_error) == size(mat_zero_mad)
    # Should contain NaN values due to zero MAD
    @test any(isnan.(result_no_error))

    # Test with zero MAD and error_on_zero_mad = true
    mat_zero_mad_error = [
        0.5 1 2 3 3.5;
        7 3 5 0 3.5;
        8 2 5 6 0
    ]

    @test_throws ErrorException(
        "The MAD (median absolute deviation) of the following " *
            "slices along dimension 2: [\"3\", \"5\"] is zero, which " *
            "implies that some of the data is very close to the " *
            "median. the data is very close to the median. Please " *
            "check your data."
    ) BigRiverJunbi.huberize(mat_zero_mad_error)
end

@testitem "huberize vector" begin
    # Test normal vector huberization
    x = [1.0, 2.0, 3.0, 4.0, 10.0]  # 10.0 is an outlier
    result = BigRiverJunbi.huberize(x)

    # Result should have same length
    @test length(result) == length(x)

    # Outlier should be reduced
    @test result[5] < x[5]

    # Non-outliers should be relatively unchanged or slightly modified
    @test result[1:4] ≈ x[1:4] rtol = 0.3

    # Test with different alpha
    result_alpha = BigRiverJunbi.huberize(x; alpha = 0.5)
    @test length(result_alpha) == length(x)
    @test result_alpha != result

    # Test error with zero MAD
    x_zero_mad = [2.0, 2.0, 2.0, 2.0]
    @test_throws ErrorException BigRiverJunbi.huberize(x_zero_mad)

    # Test no error with zero MAD when flag is false
    result_no_error = BigRiverJunbi.huberize(x_zero_mad; error_on_zero_mad = false)
    @test length(result_no_error) == length(x_zero_mad)
    @test any(isnan.(result_no_error))
end

@testitem "huberloss" begin
    # Test basic functionality
    @test BigRiverJunbi.huberloss(0.5) ≈ 0.125  # |x| ≤ α, so x²/2
    @test BigRiverJunbi.huberloss(-0.5) ≈ 0.125  # Same for negative
    @test BigRiverJunbi.huberloss(1.0) ≈ 0.5     # |x| = α, boundary case
    @test BigRiverJunbi.huberloss(-1.0) ≈ 0.5    # Same for negative

    # Test for |x| > α case
    result = BigRiverJunbi.huberloss(2.0; alpha = 1.0)
    expected = 1.0 * (2.0 - 1.0^2 / 2)  # α(|x| - α²/2) = 1.0 * (2.0 - 0.5) = 1.5
    @test result ≈ expected

    # Test with different alpha
    @test BigRiverJunbi.huberloss(1.5; alpha = 2.0) ≈ 1.5^2 / 2  # |x| ≤ α
    @test BigRiverJunbi.huberloss(3.0; alpha = 2.0) ≈ 2.0 * (3.0 - 2.0^2 / 2)  # |x| > α

    # Test error with non-positive alpha
    @test_throws AssertionError BigRiverJunbi.huberloss(1.0; alpha = 0.0)
    @test_throws AssertionError BigRiverJunbi.huberloss(1.0; alpha = -1.0)

    # Test monotonicity: loss should increase with |x|
    losses = [BigRiverJunbi.huberloss(x) for x in [0.1, 0.5, 1.0, 1.5, 2.0]]
    @test all(diff(losses) .> 0)
end

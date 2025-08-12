"""
    missing_summary(df::DataFrame)

Adds a row and column to the dataframe that contains the percentage of missing values in
each column and row. Returns a pretty table with the percentage of missing values in the
last row and column highlighted.

!!! warning
    This function will not preserve the type of the dataframe, as it converts everything
    to a string for the pretty table. It is primarily used for quick visualizations. For
    getting the actual missing percentages, use the [`missing_percentages`](@ref) function
    instead.

# Arguments
- `df`: The dataframe to add the missing summary to.

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

julia> BigRiverJunbi.missing_summary(df)
┌───────────────┬────────┬─────────┬─────────┬─────────┬─────────┬───────────────┐
│               │      A │       B │       C │       D │       E │ pmissing_rows │
│               │ String │  String │  String │  String │  String │        String │
├───────────────┼────────┼─────────┼─────────┼─────────┼─────────┼───────────────┤
│             1 │      1 │ missing │ missing │       6 │ missing │           0.6 │
│             2 │      2 │ missing │       4 │ missing │ missing │           0.6 │
│             3 │      3 │ missing │       5 │       7 │      10 │           0.2 │
├───────────────┼────────┼─────────┼─────────┼─────────┼─────────┼───────────────┤
│ pmissing_cols │    0.0 │     1.0 │    0.33 │    0.33 │    0.67 │          0.47 │
└───────────────┴────────┴─────────┴─────────┴─────────┴─────────┴───────────────┘
```
"""
function missing_summary(df::DataFrame)
    # get the missing percentages
    pmissing_cols, pmissing_rows, total_missing = missing_percentages(df)
    # convert everything to string, including the dataframe
    df = string.(df)
    pmissing_cols = string.(round.(pmissing_cols, digits = 2))
    pmissing_rows = string.(round.(pmissing_rows, digits = 2))
    total_missing = string(round(total_missing, digits = 2))
    push!(pmissing_rows, total_missing) # add a row for the total
    # add the missing counts and rows to the dataframe
    df = vcat(
        df,
        DataFrame(
            Dict(name => value for (name, value) in zip(names(df), pmissing_cols))
        )
    )
    df = insertcols(df, :pmissing_rows => pmissing_rows)
    hl_t = Highlighter(
        (data, i, j) -> i == size(df, 1) && j == size(df, 2),
        Crayon(bold = true, foreground = :green)
    )
    hl_r = Highlighter((data, i, j) -> i == size(df, 1), Crayon(foreground = :red))
    hl_c = Highlighter((data, i, j) -> j == size(df, 2), Crayon(foreground = :blue))
    nrows = nrow(df)
    hlines = [0, 1, nrows, nrows + 1]
    row_labels = vcat(1:(nrows - 1), "pmissing_cols")
    return pretty_table(
        df,
        row_labels = row_labels,
        highlighters = (hl_t, hl_c, hl_r),
        hlines = hlines
    )
end

"""
    missing_percentages(df::DataFrame)

Returns the percentage of missing values in each column and row, as well as the total
percentage of missing values in the dataframe.

# Arguments
- `df`: The dataframe to calculate the missing percentages for.

# Returns
- `pmissing_cols`: A `Vector` of the percentage of missing values in each column.
- `pmissing_rows`: A `Vector` of the percentage of missing values in each row.
- `total_missing`: The total percentage of missing values in the dataframe.

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

julia> BigRiverJunbi.missing_percentages(df)
([0.0, 1.0, 0.3333333333333333, 0.3333333333333333, 0.6666666666666666], [0.6, 0.6, 0.2], 0.4666666666666667)
```
"""
function missing_percentages(df::DataFrame)
    count_missing_cols = [count(ismissing, col) for col in eachcol(df)]
    count_missing_rows = [count(ismissing, row) for row in eachrow(df)]
    pmissing_cols = count_missing_cols ./ size(df, 1)
    pmissing_rows = count_missing_rows ./ size(df, 2)
    total_missing = (sum(count_missing_cols)) / (size(df, 1) * size(df, 2))
    return pmissing_cols, pmissing_rows, total_missing
end

# Functions for dataframes

These functions are designed to be used in conjunction with the [DataFrames.jl](https://dataframes.juliadata.org/stable/) package. Since this is a heavy dependency, DataFrames.jl is not shipped directly with BigRiverJunbi.jl. Instead, this is part of a package extension that can be loaded by installing DataFrames.jl separately and simply typing `using DataFrames` in the REPL or your code before using these functions.

## Imputation

```@autodocs
Modules = [Base.get_extension(BigRiverJunbi, :DataFramesExt)]
Pages = ["ext/DataFramesExt/impute.jl"]
```

## Normalization

```@autodocs
Modules = [Base.get_extension(BigRiverJunbi, :DataFramesExt)]
Pages = ["ext/DataFramesExt/normalize.jl"]
```

## Transformation

```@autodocs
Modules = [Base.get_extension(BigRiverJunbi, :DataFramesExt)]
Pages = ["ext/DataFramesExt/transforms.jl"]
```

## Utility functions

```@autodocs
Modules = [Base.get_extension(BigRiverJunbi, :DataFramesExt)]
Pages = ["ext/DataFramesExt/utils.jl"]
```

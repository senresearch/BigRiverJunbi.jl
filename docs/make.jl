using BigRiverJunbi
using Documenter

DocMeta.setdocmeta!(
    BigRiverJunbi,
    :DocTestSetup,
    :(using BigRiverJunbi, DataFrames);
    recursive = true
)

makedocs(;
    modules = [BigRiverJunbi],
    authors = "Abhirath Anand <74202102+theabhirath@users.noreply.github.com> and contributors",
    sitename = "BigRiverJunbi.jl",
    format = Documenter.HTML(;
        canonical = "https://senresearch.github.io/BigRiverJunbi.jl",
        edit_link = "main",
        assets = String[]
    ),
    pages = ["Home" => "index.md"]
)

deploydocs(;
    repo = "github.com/senresearch/BigRiverJunbi.jl",
    devbranch = "main",
    devurl = "stable"
)

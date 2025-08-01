using BigRiverJunbi
using DataFrames
using Documenter
using DocumenterVitepress

DocMeta.setdocmeta!(BigRiverJunbi, :DocTestSetup, :(using BigRiverJunbi); recursive = true)

DocMeta.setdocmeta!(
    isdefined(Base, :get_extension) ? Base.get_extension(BigRiverJunbi, :DataFramesExt) :
        BigRiverJunbi.DataFramesExt,
    :DocTestSetup,
    :(using BigRiverJunbi, DataFrames);
    recursive = true
)

makedocs(;
    modules = [BigRiverJunbi],
    authors = "Abhirath Anand <74202102+theabhirath@users.noreply.github.com> and contributors",
    sitename = "BigRiverJunbi.jl",
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "https://github.com/senresearch/BigRiverJunbi.jl",
    ),
    pages = [
        "Home" => "index.md",
        "API" => [
            "For matrices" => "api.md",
            "For dataframes" => "df-api.md",
        ]
    ]
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/senresearch/BigRiverJunbi.jl",
    target = "build",
    devbranch = "main",
    branch = "gh-pages",
    push_preview = true
)

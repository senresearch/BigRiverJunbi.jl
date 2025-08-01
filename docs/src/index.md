```@raw html
---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: "BigRiverJunbi.jl"
  tagline: Data Preparation for 'omics data in Julia
  actions:
    - theme: brand
      text: Get Started
      link: /get_started
    - theme: alt
      text: View on Github
      link: https://github.com/senresearch/BigRiverJunbi.jl
    - theme: alt
      text: API
      link: /api
features:
  - title: What is BigRiverJunbi.jl?
    details: BigRiverJunbi.jl is a Julia package for 'omics data preprocessing. It provides functions for data imputation, normalization, transformation and standardization.
  - title: Why the name?
    details: The word &quot;Junbi&quot; (準備, 준비, 准备) is &quot;preparation&quot; in Chinese, Korean, and Japanese (the pronunciation is slightly different in each of these).
---
```

````@raw html
<div class="vp-doc" style="width:80%; margin:auto">

<h2> Installation </h2>

You can install BigRiverJunbi using Julia's package manager `Pkg`:

```julia
julia> using Pkg

julia> Pkg.add("BigRiverJunbi")
```

Or you can use the package mode in the REPL:

```julia
] add BigRiverJunbi
```

</div>
````

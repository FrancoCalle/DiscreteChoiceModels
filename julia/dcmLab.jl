module dcmLab
    using LinearAlgebra, Combinatorics
    using Optim, Distributions
    using DataFrames
    using TexTables, OrderedCollections
    using Base.Threads
    using StatsBase
    using Plots
    using Combinatorics

    export fit
    export fit!
    export coef
    export summary
    export tidy

    ## Modules ##
    include("generate_fake_data.jl")
    include("utils.jl")
    include("elogit.jl")
    include("elogit_asc.jl")
    include("elogit_dsc.jl")
    include("elogit_hybrid.jl")
    include("logit.jl")
    include("logit_asc.jl")
    include("logit_dsc.jl")
    include("logit_hybrid.jl")

end 
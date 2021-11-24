module dcmLab
    using LinearAlgebra, Combinatorics
    using Optim, Distributions
    using DataFrames
    using TexTables, OrderedCollections
    using RDatasets
    using StatsBase
    using Plots
    using Combinatorics

    export fit
    export fit!
    export coef
    export summary
    export tidy

    ## Types ##
    # abstract type Model end
    # abstract type LinearModel <: Model  end
    # abstract type GeneralisedLinearModel <: Model end


    # abstract type Fit <: Model end
    # abstract type LinearModelFit <: Fit end


    # abstract type vcov end

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
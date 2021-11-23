using Pkg
using LinearAlgebra, Combinatorics
using Optim, Distributions
using DataFrames
using TexTables, OrderedCollections
using RDatasets
using StatsBase
using Plots


struct define_parameters
    N::Int64
    J::Int64
    T::Int64
    β::Array
    σ::Float64
    ρ::Float64
end


function compute_fake_data(parameters)

    N = parameters.N
    J = parameters.J
    T = parameters.T
    β = parameters.β

    # Get XXs and Omega
    K = length(β)
    XX_list  = [rand(Normal(),(N,J)) for i in 1:K]
    Ω = rand(Normal(),(N,J)) .> 0.7

    # Placeholder for utilities
    u = zeros(size(XX_list[1])) 
    # Add atributes
    for k in 1:K u .+= XX_list[k].*β[k] end 
    # Add logit shock
    u .+= rand(Gumbel(0, 1),(N,J))
    # Eliminate unfeasible options:
    u .= u.*Ω
    # Fill them with -Inf:
    u[.~Ω] .= -Inf
    # Get maximum:
    max_u = maximum.(eachrow(u))
    # Get probabilities:
    pr = exp.(u .- max_u)./sum.(eachrow(exp.(u .- max_u)))
    # Sample outcome based on probs:    
    # Y = [sample(1:J, Weights(pr[ii,:])) for ii in 1:N]
    Y = zeros(N); for ii in 1:N _, Y[ii]=findmax(pr[ii,:]) end

    return Y, XX_list, Ω

end


function objective_function(β, Y, XX_list, Ω)

    N = size(XX_list[1],1)      #J = size(XX_list[1],2)
    K = length(XX_list)
    
    u = zeros(size(XX_list[1])); for k in 1:K u .+= XX_list[k].*β[k] end 
    u .= u.*Ω
    u[.~Ω] .= -Inf
    max_u = maximum.(eachrow(u))

    Pr = exp.(u .- max_u)./sum.(eachrow(exp.(u .- max_u)))
    Pr_i = [Pr[ii,:][Int(Y[ii])] for ii in 1:N]
    
    logL = mean(log.(Pr_i[.~isnan.(Pr_i)]))
    
    return -logL

end


parameters = define_parameters(10000,20,5,rand(5),0.5,0.9)
Y, XX_list, Ω = compute_fake_data(parameters);
β₀ = rand(5)


func(β) = objective_function(β, Y, XX_list, Ω)
res = optimize(func, β₀, LBFGS(), Optim.Options(iterations = 1000))
β_hat = Optim.minimizer(res)
scatter(parameters.β,β_hat)


function objective()


    return

end
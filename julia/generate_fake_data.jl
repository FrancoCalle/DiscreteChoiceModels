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
    # Sample outcome based on probs:     # Y = [sample(1:J, Weights(pr[ii,:])) for ii in 1:N]
    Y = zeros(N); for ii in 1:N _, Y[ii]=findmax(pr[ii,:]) end

    return Y, XX_list, Ω

end


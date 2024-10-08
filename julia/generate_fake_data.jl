
struct define_parameters
    N::Int64
    J::Int64
    T::Int64
    β::Array
    γ::Float64
    γ0::Float64
end



function compute_fake_data(parameters)

    N = parameters.N
    J = parameters.J
    β = parameters.β

    # Get XXs and Omega
    K = length(β)
    XX_list  = [rand(Normal(),(N,J)) for i in 1:K]
    # XX_list[1] .= 1
    Ω = rand(Normal(),(N,J)) .> -Inf

    # Prices
    p = rand(Normal(),J)

    # Placeholder for utilities
    u = zeros(size(XX_list[1])) 
    # Add atributes
    for k in 1:K 
        u .+= XX_list[k].*β[k] 
    end 
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
    Y = zeros(N); 
    Rank = Array{Int64}[]
    for ii in 1:N 
        _, Y[ii]=findmax(u[ii,:]) 
        ω = findall(x->x==1, Ω[ii,:])
        u_lookup = Dict(Pair.(ω,u[ii,:][ω]))
        rank = sort(ω, by=x->u_lookup[x], rev=true)
        push!(Rank, Int.(rank))
    end
    
    Y = Int.(Y)

    return Y, XX_list, p, Rank, Ω

end


function compute_fake_data_asc(parameters)

    N = parameters.N
    J = parameters.J
    β = parameters.β

    γ = parameters.γ
    γ0 = parameters.γ0


    # Get XXs and Omega
    K = length(β)
    XX_list  = [rand(Normal(),(N,J)) for i in 1:K]

    # Consideration set variables:
    p = rand(Normal(),J)    # Prices
    x = rand(Normal(),N)    # X Characteristics:
    px = x .* p'

    v_search = γ .* px .+ γ0
    pr_search = exp.(v_search) ./ (1 .+ exp.(v_search))
    Ω = Bool.(rand.(Binomial.(Ref(1), pr_search)))

    flag_non = sum(Ω, dims=2) .== 0
    if sum(flag_non)>0
        index_non = findall(x -> x == 1 ,flag_non[:])
        Ω[index_non,1] .= 1
    end


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
    Y = zeros(N); 
    Rank = Array{Int64}[]
    for ii in 1:N 
        _, Y[ii]=findmax(pr[ii,:]) 
        ω = findall(x->x==1, Ω[ii,:])
        u_lookup = Dict(Pair.(ω,u[ii,:][ω]))
        rank = sort(ω, by=x->u_lookup[x], rev=true)
        push!(Rank, Int.(rank))
    end
    
    Y = Int.(Y)
    
    return Y, XX_list, px, p, Rank, Ω

end



# Gen fake data:
function compute_fake_data_dsc(parameters)

    N = parameters.N
    J = parameters.J
    T = parameters.T
    β = parameters.β

    # Get XXs and Omega
    K = length(β)
    XX_list  = [rand(Normal(),(N,J)) for i in 1:K]
    Ω = rand(Normal(),(N,J)) .> 0.6

    # Prices
    p = rand(Normal(),(J,T))
    Y_t = Any[]
    Rank_t = Any[]

    for t in 1:T
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
        # Get maximum: # max_u = maximum.(eachrow(u)) # Get probabilities: # pr = exp.(u .- max_u)./sum.(eachrow(exp.(u .- max_u)))
        # Sample outcome based on probs:     # Y = [sample(1:J, Weights(pr[ii,:])) for ii in 1:N]
        Y = zeros(N); 
        Rank = Array{Int64}[];
        for ii in 1:N 
            _, Y[ii]=findmax(u[ii,:])
            ω = findall(x->x==1, Ω[ii,:])
            u_lookup = Dict(Pair.(ω,u[ii,:][ω]))
            rank = sort(ω, by=x->u_lookup[x], rev=true)
            push!(Rank, Int.(rank))
        end
        push!(Rank_t, Rank)
        push!(Y_t, Int.(Y))
    end
    
    return Y_t, XX_list, p, Rank_t, Ω

end

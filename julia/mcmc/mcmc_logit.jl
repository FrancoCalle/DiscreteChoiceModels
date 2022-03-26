push!(LOAD_PATH, pwd())
using Pkg
using dcmLab
using Plots
using Optim
using LinearAlgebra
using Combinatorics
using Distributions


struct ModelData
    Y::Array
    XX_list::Array
    px::Array
    Cset_list::Array
end

struct ModelParameters
    α::Array
end

struct MarkovState
    Data::ModelData
    Θ::ModelParameters
    log_P::Float64
end


function MetropolisHastings!(m::MarkovState)

    # Unpack Data:
    data = m.Data;

    Y = data.Y
    XX_list = data.XX_list
    px = data.px
    Cset_list = data.Cset_list


    # Unpack previous params:
    Θ = m.Θ; α = Θ.α

    # Unpack previous loglikelihood
    log_P = m.log_P

    nparam = length(α)
    nparam_γ = 0
    nparam_β = nparam - nparam_γ

    β = α[1:nparam_β]


    β_star = rand(MvNormal(β, Matrix(I, nparam_β, nparam_β).*0.02))
    
    α_star = copy(β_star)

    Θ_star = ModelParameters(α_star)

    log_P_star = dcmLab.logit(α_star, Y, XX_list, Cset_list)

    dif_log_q = 0


    # Acceptance probability
    aalpha = exp(log_P_star)/exp(log_P) # Start with posterior distribution contribution ...
        
    aalpha += dif_log_q # print("acceptance probability:" , aalpha, "\n")

    if rand(Uniform(0,1)) < min(1,aalpha)
        
        m_prime = MarkovState(data, Θ_star, log_P_star)

    else

        m_prime = m
        
    end

    return m_prime
    
end



function MetropolisHastings(nIter::Int,      # Number of iterations
                            m_draw::MarkovState)

    α_draws = zeros(nIter,5)

    for ii in 1:nIter
        
        m_draw = MetropolisHastings!(m_draw);
        
        α_draws[ii,:] = m_draw.Θ.α
    
    end

    return α_draws
end


#-----------------------------------------------------------------------------------

# Parameter Values:
R = 1000
N = 5000;
J = 10;
T = 5;

β = [0.972, 0.67, 0.63, 0.55, 0.98];
γ = 2.5;
γ0 = .01;

# 
θ_true = zeros(7)
θ_true[1:5] = β
θ_true[6] = γ0
θ_true[7] = γ

# Check number of threads to be used:
Threads.nthreads()


# ASC: Optimizing Well
#------------------------------------------------------ 

# Set Parameters and Simulate Fake Data:

parameters = dcmLab.define_parameters(N,J,T,β,γ,γ0);
Y, XX_list, p, Rank, Ω = dcmLab.compute_fake_data(parameters);

α_init = β #rand(7) # α_init = rand(7)
md = ModelData(Y, XX_list, p, Ω);
Θ_init = ModelParameters(α_init); # Initialize parameters
m_init = MarkovState(md, Θ_init, dcmLab.logit(α_init, Y, XX_list, Ω)); # Initialize markov state
α_draws = MetropolisHastings(R, m_init);

histogram(α_draws[:,2])

plot(α_draws[:,1])





##########################################################

N = size(XX_list[1],1)
K = length(XX_list)

u = zeros(size(XX_list[1])); for k in 1:K u .+= XX_list[k].*β[k] end 
u .= u.*Ω
u[.~Ω] .= -Inf
max_u = 0 #maximum.(eachrow(u))

Pr = exp.(u .- max_u) ./ (1 .+ sum.(eachrow(exp.(u .- max_u))))
Pr_i = [Pr[ii,:][Int(Y[ii])] for ii in 1:N]

logL = mean(log.(Pr_i[.~(Pr_i.==0)]))



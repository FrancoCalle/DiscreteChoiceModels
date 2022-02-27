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

    # Propose ...

    pp = 1 #sample(1:2)

    if pp == 1 # Sample β

        α_star = rand(MvNormal(α, Matrix(I, nparam, nparam).*0.05))
        
        Θ_star = ModelParameters(α_star)
    
        log_P_star = dcmLab.logit_asc(α_star, Y, XX_list, px, Cset_list)

        dif_log_q = 0

    end


    # if pp == 2 # Sample τ

    #     β_star = β
    
    #     τ2_star = exp(rand(Normal(log(τ2), c))) # rand(Gamma(τ2*c, c))
        
    #     α_i_star = rand(Normal(0, τ2_star), data.N, S)
    
    #     Θ_star = ModelParameters(β_star, τ2_star, α_i_star)
    
    #     log_P_star = log_target_pdf(data, Θ_star)

    #     dif_log_q = 0

    # end

    
    # Acceptance probability
    aalpha = log_P_star - log_P # Start with posterior distribution contribution ...
        
    aalpha += dif_log_q # print("acceptance probability:" , aalpha, "\n")

    if log(rand(Uniform(0,1))) < min(0,aalpha)
        
        m_prime = MarkovState(data, Θ_star, log_P_star)

    else

        m_prime = m
        
    end

    return m_prime
    
end



function MetropolisHastings(nIter::Int,      # Number of iterations
                            m_draw::MarkovState)

    α_draws = zeros(nIter,7)

    for ii in 1:nIter
        
        m_draw = MetropolisHastings!(m_draw);
        
        α_draws[ii,:] = m_draw.Θ.α
    
    end

    return α_draws
end


#-----------------------------------------------------------------------------------


# Parameter Values:
N = 300;
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
Y, XX_list, px, p, Rank, Ω = dcmLab.compute_fake_data_asc(parameters);
Cset_list = dcmLab.obtain_power_set(J,Y);


α_init = copy(θ_true) #rand(7)
md = ModelData(Y, XX_list, px, Cset_list);
Θ_init = ModelParameters(α_init); # Initialize parameters
m_init = MarkovState(md, Θ_init, dcmLab.logit_asc(α_init, Y, XX_list, px, Cset_list)); # Initialize markov state
α_draws = MetropolisHastings(2000, m_init);

histogram(α_draws[:,1])

plot(α_draws[:,1])


nIter= 300
m_draw = m_init
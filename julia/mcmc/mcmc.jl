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


logistic(p, γ0, γ) = exp(γ0 + p*γ)/(1 + exp(γ0 + p*γ))


function choiceProbability_ij(C, u_i, j)

    u_j = u_i[j]
    u_C = u_i[C]
    Pr_j = exp(u_j)/sum(exp.(u_C))

    return Pr_j
end

function considerationProbability_C(C, Ω_i, j, p, γ0, γ)

    π_C = 1
    for c in Ω_i
        if c in C
            π_C *= logistic(p[c], γ0, γ)
        else
            π_C *= (1-logistic(p[c], γ0, γ))
        end
    end

    return π_C

end


function prob_asc_ij(Ω_i, Cset, u_i, j, p, γ0, γ)

    π_C = considerationProbability_C.(Cset, Ref(Ω_i), j, Ref(p), γ0, γ)
    s_C = choiceProbability_ij.(Cset, Ref(u_i), j)
    s_ij = sum(π_C.*s_C)

    return s_ij

end

function obtain_power_set(J,Y)

    Ω_i = Array(1:J)

    Cset_list = [P(Ω_i,Y[j]) for j = 1:size(Y,1)]; # Power set for all individuals...
    
    return Cset_list

end


function logit_asc(θ, Y, XX_list, PP, Cset_list)

    N = size(XX_list[1],1)
    K = length(XX_list)
    J = size(XX_list[1],2)

    β = θ[1:K]
    γ0 = θ[K+1]
    γ = θ[K+2]

    u = zeros(size(XX_list[1])); for k in 1:K u .+= XX_list[k].*β[k] end 
    
    logL = 0

    for ii = 1:N
        j = Y[ii]       # Outcome selected
        Ω_i = Array(1:J) #Rank[ii]  # 
        Cset = Cset_list[ii]
        u_i = u[ii,:]
        p = PP[ii,:]
        sj = prob_asc_ij(Ω_i, Cset, u_i, j, p, γ0, γ)
        if sj>0
            logL += log(sj)/N
        end
    end

    return -logL

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

        α_star = rand(MvNormal(α, Matrix(I, nparam, nparam).*0.2))
        
        Θ_star = ModelParameters(α_star)
    
        log_P_star = -logit_asc(α_star, Y, XX_list, px, Cset_list)

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
m_init = MarkovState(md, Θ_init, -logit_asc(α_init, Y, XX_list, px, Cset_list)); # Initialize markov state
α_draws = MetropolisHastings(300, m_init);

histogram(α_draws[:,1])

plot(α_draws[:,1])


nIter= 300
m_draw = m_init



logL_list = zeros(size(α_draws,1))
for ii = 1:size(α_draws,1)
    logL_list[ii] = dcmLab.logit_asc(α_draws[ii,:], Y, XX_list, px, Cset_list)
end


plot(logL_list)

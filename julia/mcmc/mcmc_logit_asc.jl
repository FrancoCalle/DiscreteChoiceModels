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
    nparam_γ = 2
    nparam_β = length(α) - nparam_γ

    β = α[1:nparam_β]
    γ0 = α[nparam_β+1]
    γ1 = α[nparam_β+2]

    # Propose ...

    pp = sample(1:3)

    α_star = copy(α)

    if pp == 1 # Sample β

        β_star = rand(MvNormal(β, Matrix(I, nparam_β, nparam_β).*0.02))
        
        α_star[1:nparam_β] = β_star

        Θ_star = ModelParameters(α_star)
    
        log_P_star = -dcmLab.logit_asc(α_star, Y, XX_list, px, Cset_list)

        dif_log_q = 0

    end


    if pp == 2 # Sample γ0

        γ0_star = rand(Normal(γ0, 0.01))
        
        α_star[nparam_β+1] = γ0_star

        Θ_star = ModelParameters(α_star)
    
        log_P_star = -dcmLab.logit_asc(α_star, Y, XX_list, px, Cset_list)

        dif_log_q = 0

    end


    if pp == 3 # Sample γ1

        γ1_star = rand(Normal(γ1, 0.01))
        
        α_star[nparam_β+2] = γ1_star

        Θ_star = ModelParameters(α_star)
    
        log_P_star = -dcmLab.logit_asc(α_star, Y, XX_list, px, Cset_list)

        dif_log_q = 0

    end


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

    α_draws = zeros(nIter,7)

    for ii in 1:nIter
        
        m_draw = MetropolisHastings!(m_draw);
        
        α_draws[ii,:] = m_draw.Θ.α
    
    end

    return α_draws
end


#-----------------------------------------------------------------------------------

# Parameter Values:
R = 5000
N = 5000;
J = 4;
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


α_init = copy(θ_true) #rand(7) # α_init = rand(7)
md = ModelData(Y, XX_list, px, Cset_list);
Θ_init = ModelParameters(α_init); # Initialize parameters
m_init = MarkovState(md, Θ_init, -dcmLab.logit_asc(α_init, Y, XX_list, px, Cset_list)); # Initialize markov state
α_draws = MetropolisHastings(R, m_init);

histogram(α_draws[:,2])

plot(α_draws[:,4])


nIter= 300
m_draw = m_init



logL_list = zeros(size(α_draws,1))
for ii = 1:size(α_draws,1)
    logL_list[ii] = exp(-dcmLab.logit_asc(α_draws[ii,:], Y, XX_list, px, Cset_list))
end


plot(logL_list)

flag = rand(4)
flag_star= rand(MvNormal(flag, Matrix(I, 4, 4).*0.02))

pdf(MvNormal(flag, Matrix(I, 4, 4).*0.02), flag_star) - pdf(MvNormal(flag_star, Matrix(I, 4, 4).*0.02), flag)

push!(LOAD_PATH, pwd())
using Pkg
using dcmLab
using Plots
using Optim
using Combinatorics
using Distributions


# Initial Values:
β₀ = rand(5)

# Logit:
#------------------------------------------------------

# Set Parameters and Simulate Fake Data:
parameters = dcmLab.define_parameters(10000,10,5,rand(5),0.5,0.9)
Y, XX_list, p, Rank, Ω = dcmLab.compute_fake_data(parameters);

# Optimize using built in solvers:
func(β) = dcmLab.logit(β, Y, XX_list, Ω)
res = optimize(func, β₀, LBFGS(), Optim.Options(iterations = 1000))
β_hat = Optim.minimizer(res)

# Plot Fit:
scatter(parameters.β,β_hat)

# Exploded Logit:
#------------------------------------------------------

# Set Parameters and Simulate Fake Data:
parameters = dcmLab.define_parameters(10000,10,5,rand(5),0.5,0.9)
Y, XX_list, p, Rank, Ω = dcmLab.compute_fake_data(parameters);

β₀ = rand(5)
θ_0 = zeros(7)
θ_0[1:5] = β₀ 
θ_0[6] = -0.3
θ_0[7] = 0.5

func(β) = dcmLab.elogit(β, Rank, XX_list, Ω)
res = optimize(func, β₀, LBFGS(), Optim.Options(iterations = 1000))
β_hat = Optim.minimizer(res)

# Plot Fit:
scatter(parameters.β,β_hat)



# logit_asc: Optimizing Well
#------------------------------------------------------

# Set Parameters and Simulate Fake Data:
parameters = dcmLab.define_parameters(10000,10,5,rand(5),0.5,0.9)
Y, XX_list, p, Rank, Ω = dcmLab.compute_fake_data(parameters);

β₀ = rand(5)
θ_0 = zeros(7)
θ_0[1:5] = β₀ 
θ_0[6] = -0.3
θ_0[7] = 0.5

dcmLab.logit_asc(θ_0, Y, XX_list, p, Rank)
func(θ) = dcmLab.logit_asc(θ, Y, XX_list, p, Rank)
res = optimize(func, θ_0, LBFGS(), Optim.Options(iterations = 1000))
θ_hat = Optim.minimizer(res)

scatter(parameters.β, θ_hat[1:5])


# logit_dsc: Optimizing Well
#-------------------------------------------------------

parameters = dcmLab.define_parameters(10000,10,5,rand(5),0.5,0.9)
Y_t, XX_list, p_t, Rank_t, Ω = dcmLab.compute_fake_data_dsc(parameters)


β₀ = rand(5)
θ_0 = zeros(7)
θ_0[1:5] = β₀ 
θ_0[6] = -0.3
θ_0[7] = 0.5

dcmLab.logit_dsc(θ_0, Y_t, XX_list, p_t, Rank_t)
func(θ) = dcmLab.logit_dsc(θ, Y_t, XX_list, p_t, Rank_t)
res = optimize(func, θ_0, LBFGS(), Optim.Options(iterations = 1000))
θ_hat = Optim.minimizer(res)
scatter(parameters.β, θ_hat[1:5])


# logit_hybrid: 
#-------------------------------------------------------

parameters = dcmLab.define_parameters(10000,10,5,rand(5),0.5,0.9)
Y_t, XX_list, p_t, Rank_t, Ω = dcmLab.compute_fake_data_dsc(parameters)


logistic(p, γ0, γ) = exp(γ0 + p*γ)/(1 + exp(γ0 + p*γ))

P(Ω_i,j) = [C for C in collect(powerset(Ω_i)) if j in C]    

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

function prob_dsc_ij(Ω_i, u_i, j, j_lag, p, γ0, γ)

    μ = logistic(p[j_lag], γ0, γ)

    if j == j_lag
        s_ij = (1-μ) + μ * choiceProbability_ij(Ω_i, u_i, j)
    else
        s_ij = μ * choiceProbability_ij(Ω_i, u_i, j)
    end

    return s_ij

end

function prob_asc_ij(Ω_i, u_i, j, p, γ0, γ)

    Cset = P(Ω_i,j)
    
    π_C = considerationProbability_C.(Cset, Ref(Ω_i), j, Ref(p), γ0, γ)
    
    s_C = choiceProbability_ij.(Cset, Ref(u_i), j)
    
    s_ij = sum(π_C.*s_C)

    return s_ij

end


function prob_hybrid_ij(Ω_i, u_i, j, j_lag, p, γ0, γ)

    μ = logistic(p[j_lag], γ0, γ)

    if j == j_lag
        s_ij = (1-μ) + μ * prob_asc_ij(Ω_i, u_i, j, p, γ0, γ)
    else
        s_ij = μ * prob_asc_ij(Ω_i, u_i, j, p, γ0, γ)
    end

    return s_ij

end


function logit_hybrid(θ, Y_t, XX_list, p_t, Rank_t)

    N = size(XX_list[1],1)
    K = length(XX_list)
    T = length(Y_t)

    β = θ[1:K]
    γ0 = θ[K+1]
    γ = θ[K+2]

    logL = 0

    for t in 2:T

        # Compute utility:
        u = zeros(size(XX_list[1])); for k in 1:K u .+= XX_list[k].*β[k] end 

        y = Y_t[t]
        rank = Rank_t[t]
        yLag = Y_t[t-1]
        p = p_t[:,t]

        for ii in 1:1000

            j = y[ii]
            j_lag = yLag[ii]
            Ω_i = rank[ii]
            u_i = u[ii,:]
            sj = prob_hybrid_ij(Ω_i, u_i, j, j_lag, p, γ0, γ)

            # println(sj)
            if (sj>0) & (sj<1)
                logL += log(sj)/(N*(T-1))
            end

        end

    end

    return -logL

end



logit_hybrid(θ_0, Y_t, XX_list, p_t, Rank_t)

func(θ) = logit_hybrid(θ, Y_t, XX_list, p_t, Rank_t)

res = optimize(func, θ_0, LBFGS(), Optim.Options(iterations = 1000))

θ_hat = Optim.minimizer(res)

scatter(parameters.β, θ_hat[1:5])


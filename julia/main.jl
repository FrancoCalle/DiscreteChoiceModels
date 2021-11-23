push!(LOAD_PATH, pwd())
using Pkg
using dcmLab
using Plots
using Optim
using Combinatorics

# Set Parameters and Simulate Fake Data:
parameters = dcmLab.define_parameters(10000,10,5,rand(5),0.5,0.9)
Y, XX_list, p, Rank, Ω = dcmLab.compute_fake_data(parameters);

# Initial Values:
β₀ = rand(5)

# Logit:
# Optimize using built in solvers:
func(β) = dcmLab.logit(β, Y, XX_list, Ω)
res = optimize(func, β₀, LBFGS(), Optim.Options(iterations = 1000))
β_hat = Optim.minimizer(res)

# Plot Fit:
scatter(parameters.β,β_hat)

# Exploded Logit:
# Optimize using built in solvers:
func(β) = dcmLab.elogit(β, Rank, XX_list, Ω)
res = optimize(func, β₀, LBFGS(), Optim.Options(iterations = 1000))
β_hat = Optim.minimizer(res)

# Plot Fit:
scatter(parameters.β,β_hat)



# logit_asc:
#------------------------------------------------------

logistic(p, γ0, γ) = exp(γ0 + p*γ)/(1 + exp(γ0 + p*γ))

P(Ω_i,j) = [C for C in collect(powerset(Ω_i)) if j in C]    


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

function choiceProbability_ij(C, u_i, j)

    u_j = u_i[j]
    u_C = u_i[C]
    Pr_j = exp(u_j)/sum(exp.(u_C))

    return Pr_j

end

function prob_ij(Ω_i, u_i, j, p, γ0, γ)

    Cset = P(Ω_i,j)
    π_C = considerationProbability_C.(Cset, Ref(Ω_i), j, Ref(p), γ0, γ)
    s_C = choiceProbability_ij.(Cset, Ref(u_i), j)
    s_ij = sum(π_C.*s_C)

    return s_ij

end

function logit_asc(θ, Y, XX_list, p, Rank)

    β = θ[1:5]
    γ0 = θ[6]
    γ = θ[7]

    N = size(XX_list[1],1)
    K = length(XX_list)
    u = zeros(size(XX_list[1])); for k in 1:K u .+= XX_list[k].*β[k] end 

    logL = 0

    for ii in 1:N
        j = Y[ii]
        Ω_i = Rank[ii]
        u_i = u[ii,:]
        sj = prob_ij(Ω_i, u_i, j, p, γ0, γ)
        if sj>0
            logL += log(sj)/N
        end
    end

    return -logL

end

β₀ = rand(5)
θ_0 = zeros(7)
θ_0[1:5] = β₀ 
θ_0[6] = -0.3
θ_0[7] = 0.5

logit_asc(θ_0, Y, XX_list, p, Rank)
func(θ) = logit_asc(θ, Y, XX_list, p, Rank)
res = optimize(func, θ_0, LBFGS(), Optim.Options(iterations = 1000))
θ_hat = Optim.minimizer(res)



# Using Autodiff:
od = OnceDifferentiable(func, θ_0; autodiff = :forward)
td = TwiceDifferentiable(func, θ_0; autodiff = :forward)
res = Optim.minimizer(optimize(td, θ_0, BFGS()))

# Using Autodiff:
using ForwardDiff
d_alogit(x) = ForwardDiff.gradient(func, θ_0)

push!(LOAD_PATH, pwd())
using Pkg
using dcmLab
using Plots
using Optim
using Combinatorics
using Distributions

# Initial Values:
N = 1000;
J = 10;
T = 5;
β = rand(5);
γ = .05;
γ0 = .1;

# Logit:
#------------------------------------------------------

# Set Parameters and Simulate Fake Data:
parameters = dcmLab.define_parameters(N,J,T,β,γ,γ0)
Y, XX_list, p, Rank, Ω = dcmLab.compute_fake_data(parameters);

# Optimize using built in solvers:
func(β) = dcmLab.logit(β, Y, XX_list, Ω)
res = optimize(func, rand(5), LBFGS(), Optim.Options(iterations = 1000))
β_hat = Optim.minimizer(res)

# Plot Fit:
scatter(parameters.β[1:end],β_hat[1:end])
plot!([minimum(parameters.β), maximum(parameters.β)],[minimum(parameters.β), maximum(parameters.β)])


# Exploded Logit:
#------------------------------------------------------

# Set Parameters and Simulate Fake Data:
parameters = dcmLab.define_parameters(N,J,T,β,γ,γ0)
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
plot!([minimum(parameters.β), maximum(parameters.β)],[minimum(parameters.β), maximum(parameters.β)])


# ASC: Optimizing Well
#------------------------------------------------------

# Set Parameters and Simulate Fake Data:
Threads.nthreads()

parameters = dcmLab.define_parameters(N,J,T,β,γ,γ0);
Y, XX_list, px, p, Rank, Ω = dcmLab.compute_fake_data_asc(parameters);

β₀ = rand(5)
θ_0 = zeros(7)
θ_0[1:5] = β₀ 
θ_0[6] = -0.3
θ_0[7] = 0.5

dcmLab.logit_asc(θ_0, Y, XX_list, px)
func(θ) = dcmLab.logit_asc(θ, Y, XX_list, px)
res = optimize(func, θ_0, NelderMead(), Optim.Options(iterations = 5))
θ_hat = Optim.minimizer(res)

scatter(parameters.β, θ_hat[1:5],legend=:bottomright)
plot!([minimum(parameters.β), maximum(parameters.β)],[minimum(parameters.β), maximum(parameters.β)])


# DSC: Optimizing well...
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
plot!([minimum(parameters.β), maximum(parameters.β)],[minimum(parameters.β), maximum(parameters.β)])



# logit_hybrid: Optimizing well...
#-------------------------------------------------------

parameters = dcmLab.define_parameters(10000,10,5,rand(5),0.5,0.9)
Y_t, XX_list, p_t, Rank_t, Ω = dcmLab.compute_fake_data_dsc(parameters)

# Random Initial Values
β0 = rand(5)
θ_0 = zeros(9)
θ_0[1:5] = β0 
θ_0[6] = -0.3
θ_0[7] = 0.5
θ_0[6] = -0.2
θ_0[7] = 0.8


dcmLab.logit_hybrid(θ_0, Y_t, XX_list, p_t, Rank_t)

func(θ) = dcmLab.logit_hybrid(θ, Y_t, XX_list, p_t, Rank_t)

res = optimize(func, θ_0, LBFGS(), Optim.Options(iterations = 1000))

θ_hat = Optim.minimizer(res)

scatter(parameters.β, θ_hat[1:5])
plot!([minimum(parameters.β), maximum(parameters.β)],[minimum(parameters.β), maximum(parameters.β)])

scatter(parameters.β, θ_hat[1:5])
plot!([minimum(parameters.β), maximum(parameters.β)],[minimum(parameters.β), maximum(parameters.β)])






# ASC: Alternative Specific Model:
#---------------------------------
using Base.Threads



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


function prob_asc_ij(Ω_i, u_i, j, p, γ0, γ)

    Cset = P(Ω_i,j)
    π_C = considerationProbability_C.(Cset, Ref(Ω_i), j, Ref(p), γ0, γ)
    s_C = choiceProbability_ij.(Cset, Ref(u_i), j)
    s_ij = sum(π_C.*s_C)

    return s_ij

end

function logit_asc(θ, Y, XX_list, pp)

    N = size(XX_list[1],1)
    J = size(XX_list[1],2)
    K = length(XX_list)

    β = θ[1:K]
    γ0 = θ[K+1]
    γ = θ[K+2]

    u = zeros(size(XX_list[1])); for k in 1:K u .+= XX_list[k].*β[k] end 
    
    logL = 0

    @threads for ii in 1:N
        j = Y[ii]
        Ω_i = Array(1:J)
        u_i = u[ii,:]
        p = pp[ii,:]
        sj = prob_asc_ij(Ω_i, u_i, j, p, γ0, γ)
        if sj>0
            logL += log(sj)/N
        end
    end

    return -logL

end





#---------------------------------------------------------------------

β0 = rand(5)
θ_0 = zeros(9)
θ_0[1:5] = β0 
θ_0[6] = -0.3
θ_0[7] = 0.5
θ_0[6] = -0.2
θ_0[7] = 0.8



θ_true = zeros(7)
θ_true[1:5] = β
θ_true[6] = γ0
θ_true[7] = γ


Threads.nthreads()

parameters = dcmLab.define_parameters(N,J,T,β,γ,γ0);
Y, XX_list, px, p, Rank, Ω = dcmLab.compute_fake_data_asc(parameters);
sum(sum(Ω, dims=2) .== 0)

logit_asc(θ_0, Y, XX_list, px)


#Optimization
logit_asc(θ_0, Y, XX_list, px)

func(θ) = logit_asc(θ, Y, XX_list, px)

res = optimize(func, θ_0, NelderMead(), Optim.Options(iterations = 2000))

θ_hat = Optim.minimizer(res)

scatter(θ_true, θ_hat,legend=:bottomright)
plot!([minimum(parameters.β), maximum(parameters.β)],[minimum(parameters.β), maximum(parameters.β)])


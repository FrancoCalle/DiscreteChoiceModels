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


# logit_dsc: Optimizing well...
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



# logit_hybrid: Optimizing well...
#-------------------------------------------------------

parameters = dcmLab.define_parameters(10000,10,5,rand(5),0.5,0.9)
Y_t, XX_list, p_t, Rank_t, Ω = dcmLab.compute_fake_data_dsc(parameters)


β₀ = rand(5)
θ_0 = zeros(9)
θ_0[1:5] = β₀ 
θ_0[6] = -0.3
θ_0[7] = 0.5
θ_0[6] = -0.2
θ_0[7] = 0.8

dcmLab.logit_hybrid(θ_0, Y_t, XX_list, p_t, Rank_t)

func(θ) = dcmLab.logit_hybrid(θ, Y_t, XX_list, p_t, Rank_t)

res = optimize(func, θ_0, LBFGS(), Optim.Options(iterations = 1000))

θ_hat = Optim.minimizer(res)

scatter(parameters.β, θ_hat[1:5])


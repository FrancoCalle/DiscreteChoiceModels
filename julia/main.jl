push!(LOAD_PATH, pwd())
using Pkg
using dcmLab
using Plots
using Optim

# Set Parameters and Simulate Fake Data:
parameters = dcmLab.define_parameters(10000,20,5,rand(5),0.5,0.9)
Y, XX_list, Rank, Ω = dcmLab.compute_fake_data(parameters);

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

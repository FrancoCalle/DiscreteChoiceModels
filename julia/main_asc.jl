push!(LOAD_PATH, pwd())
using Pkg
using dcmLab
using Plots
using Optim
using Combinatorics
using Distributions

# Parameter Values:
N = 1000;
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

# Set Initial Parameters
β0 = rand(5)
θ_0 = zeros(7)
θ_0[1:5] = β0
θ_0[6] = -0.3
θ_0[7] = 0.5

# Check if function working well:
dcmLab.logit_asc(θ_0, Y, XX_list, px)

# Optimize:
func(θ) = dcmLab.logit_asc(θ, Y, XX_list, px)
res = optimize(func, θ_true, NelderMead(), Optim.Options(iterations = 1500))
θ_hat = Optim.minimizer(res)

scatter(θ_true, θ_hat,legend=:bottomright)
plot!([minimum(θ_true), maximum(θ_true)],[minimum(θ_true), maximum(θ_true)])


# Grid Search over different parameters:
# -------------------------------------

γ_grid = Array(γ- 5:0.05:γ+ 5)
obj_list = zeros(length(γ_grid))
for ii in 1:length(γ_grid)
    parameters = θ_true
    parameters[end] = γ_grid[ii]
    obj_list[ii] = dcmLab.logit_asc(parameters, Y, XX_list, px)
end

plot(γ_grid,obj_list)
plot(γ_grid, obj_list, linewidth = 5, linecolor=:red, label="Log Likelihood" )
vline!([γ], linewidth=4, linecolor=:blue, label="α = 1")
savefig("q2_minimizing_at_alpha.pdf")

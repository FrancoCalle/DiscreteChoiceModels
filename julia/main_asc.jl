push!(LOAD_PATH, pwd())
using Pkg
using dcmLab
using Plots
using Optim
using Combinatorics
using Distributions

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

# Set Initial Parameters
β0 = rand(5)
θ_0 = zeros(7)
θ_0[1:5] = β0
θ_0[6] = -0.3
θ_0[7] = 0.5

# Check if function working well:
dcmLab.logit_asc(θ_0, Y, XX_list, px, Cset_list)

# Optimize:
func(θ) = dcmLab.logit_asc(θ, Y, XX_list, px, Cset_list)
res = optimize(func, θ_true, NelderMead(), Optim.Options(iterations = 3000,
                                                        show_trace=true,
                                                        show_every=20)
                                                        )
θ_hat = Optim.minimizer(res)

scatter(θ_true, θ_hat,legend=:bottomright)
plot!([minimum(θ_true), maximum(θ_true)],[minimum(θ_true), maximum(θ_true)])


# Grid Search over different parameters:
# -------------------------------------

# Minimizing at γ:
γ_grid = Array(γ- 5:0.05:γ+ 5)
obj_list = zeros(length(γ_grid))
for ii in 1:length(γ_grid)
    parameters = copy(θ_true)
    parameters[end] = γ_grid[ii]
    obj_list[ii] = dcmLab.logit_asc(parameters, Y, XX_list, px, Cset_list)
end

plot(γ_grid, obj_list, linewidth = 5, linecolor=:red, label="Log Likelihood" )
vline!([γ], linewidth=4, linecolor=:blue, label="γ = 2.5")
savefig("../figures/F1_grid_search_for_gamma.pdf")



# Minimizing at γ0:
γ0_grid = Array(γ0 - 3:0.05:γ0 + 3)
obj_list = zeros(length(γ0_grid))
for ii in 1:length(γ0_grid)
    parameters = copy(θ_true)
    parameters[end-1] = γ0_grid[ii]
    obj_list[ii] = dcmLab.logit_asc(parameters, Y, XX_list, px, Cset_list)
end

plot(γ0_grid, obj_list[1:121], linewidth = 5, linecolor=:red, label="Log Likelihood" )
vline!([γ0], linewidth=4, linecolor=:blue, label="γ = 0.01")
savefig("../figures/F1_grid_search_for_gamma0.pdf")



# Minimizing at β:
β1_grid = Array(β[1] - 3:0.05:β[1] + 3)
obj_list = zeros(length(β1_grid))
for ii in 1:length(β1_grid)
    parameters = copy(θ_true)
    parameters[1] = β1_grid[ii]
    obj_list[ii] = dcmLab.logit_asc(parameters, Y, XX_list, px, Cset_list)
end

plot(β1_grid, obj_list, linewidth = 5, linecolor=:red, label="Log Likelihood" )
vline!([β[1]], linewidth=4, linecolor=:blue, label="β1 = 0.972")
savefig("../figures/F1_grid_search_for_beta1.pdf")



# Minimizing at β2:
β2_grid = Array(β[2] - 3:0.05:β[2] + 3)
obj_list = zeros(length(β2_grid))
for ii in 1:length(β2_grid)
    parameters = copy(θ_true)
    parameters[2] = β2_grid[ii]
    obj_list[ii] = dcmLab.logit_asc(parameters, Y, XX_list, px, Cset_list)
end

plot(β2_grid, obj_list, linewidth = 5, linecolor=:red, label="Log Likelihood" )
vline!([β[2]], linewidth=4, linecolor=:blue, label="β2 = 0.67")
savefig("../figures/F1_grid_search_for_beta2.pdf")



# Minimizing at β3:
β3_grid = Array(β[3] - 3:0.05:β[3] + 3)
obj_list = zeros(length(β3_grid))
for ii in 1:length(β3_grid)
    parameters = copy(θ_true)
    parameters[3] = β3_grid[ii]
    obj_list[ii] = dcmLab.logit_asc(parameters, Y, XX_list, px, Cset_list)
end

plot(β3_grid, obj_list, linewidth = 5, linecolor=:red, label="Log Likelihood" )
vline!([β[3]], linewidth=4, linecolor=:blue, label="β3 = 0.63")
savefig("../figures/F1_grid_search_for_beta3.pdf")



# Minimizing at β5:
β4_grid = Array(β[4] - 3:0.05:β[4] + 3)
obj_list = zeros(length(β4_grid))
for ii in 1:length(β4_grid)
    parameters = copy(θ_true)
    parameters[4] = β4_grid[ii]
    obj_list[ii] = dcmLab.logit_asc(parameters, Y, XX_list, px, Cset_list)
end

plot(β4_grid, obj_list, linewidth = 5, linecolor=:red, label="Log Likelihood" )
vline!([β[4]], linewidth=4, linecolor=:blue, label="β4 = 0.98")
savefig("../figures/F1_grid_search_for_beta4.pdf")




# Minimizing at β5:
β5_grid = Array(β[5] - 3:0.05:β[5] + 3)
obj_list = zeros(length(β5_grid))
for ii in 1:length(β5_grid)
    parameters = copy(θ_true)
    parameters[5] = β5_grid[ii]
    obj_list[ii] = dcmLab.logit_asc(parameters, Y, XX_list, px, Cset_list)
end

plot(β5_grid, obj_list, linewidth = 5, linecolor=:red, label="Log Likelihood" )
vline!([β[5]], linewidth=4, linecolor=:blue, label="β5 = 0.98")
savefig("../figures/F1_grid_search_for_beta5.pdf")



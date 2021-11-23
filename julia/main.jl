push!(LOAD_PATH, pwd())
using Pkg
using dcmLab

parameters = dcmLab.define_parameters(10000,20,5,rand(5),0.5,0.9)
Y, XX_list, Ω = compute_fake_data(parameters);
β₀ = rand(5)


func(β) = objective_function(β, Y, XX_list, Ω)
res = optimize(func, β₀, LBFGS(), Optim.Options(iterations = 1000))
β_hat = Optim.minimizer(res)
scatter(parameters.β,β_hat)

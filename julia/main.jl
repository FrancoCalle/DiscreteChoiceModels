push!(LOAD_PATH, pwd())
using Pkg
using dcmLab

# Set Parameters and Simulate Fake Data:
parameters = dcmLab.define_parameters(10000,20,5,rand(5),0.5,0.9)
Y, XX_list, Ω = dcmLab.compute_fake_data(parameters);

# Initial Values:
β₀ = rand(5)

# Optimize using built in solvers:
func(β) = objective_function(β, Y, XX_list, Ω)
res = optimize(func, β₀, LBFGS(), Optim.Options(iterations = 1000))
β_hat = Optim.minimizer(res)

# Plot Fit:
scatter(parameters.β,β_hat)
β = rand(5)


function elogit(β, Y, Rank, XX_list, Ω)

    N = size(XX_list[1],1)
    K = length(XX_list)
    
    u = zeros(size(XX_list[1])); for k in 1:K u .+= XX_list[k].*β[k] end 
    u .= u.*Ω

    logL = 0

    for ii in 1:N
        
        # Get feasible set:
        ω = findall(x->x==1, Ω[ii,:])
        u_i = u[ii,:]
        u_if = sort(u_i[ω], rev=true)
        for jj in 1:length(ω)
            logL += log(exp.(u_if[jj])./sum(exp.(u_if[jj:end])))/N
        end
    end
    
    return -logL

end
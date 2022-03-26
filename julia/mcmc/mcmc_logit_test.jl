push!(LOAD_PATH, pwd())
using Pkg
using dcmLab
using Plots
using Optim
using LinearAlgebra
using Combinatorics
using Distributions
using ForwardDiff

# First set true data generating process ... 
# ------------------------------------------

# Parameter Values:
R = 1000
N = 500;
J = 10;
T = 3;

β = [1, 2, 3, 4, 5];
β_true = copy(β)
γ = 2.5;
γ0 = .01;

θ_true = zeros(7)
θ_true[1:5] = β
θ_true[6] = γ0
θ_true[7] = γ

# Compute DGP:

parameters = dcmLab.define_parameters(N,J,T,β, γ,γ0);

Y, XX_list, p, Rank, Ω = dcmLab.compute_fake_data(parameters);


# Now, define logit model and hessian which we will use for the posterior distrib:
# -------------------------------------------------------------------------------

function logit(β; Y=Y, XX_list = XX_list, Ω = Ω)

    N = length(Y)
    K = length(XX_list)
    J = maximum(Y)

    Choice = Array(1:J)' .== Y 
    
    u = zeros(size(XX_list[1])); 
    
    u = (XX_list[1].*β[1] .+ 
        XX_list[2].*β[2] .+ 
        XX_list[3].*β[3] .+ 
        XX_list[4].*β[4] .+
        XX_list[5].*β[5])

    u .= u.*Ω

    Pr = exp.(u)./ (1 .+ sum(exp.(u), dims=2))
 
    Pr_i = sum(Pr.*Choice, dims=2)

    logL = mean(log.(Pr_i))
    
    return logL

end


h = x -> .-ForwardDiff.hessian(logit, x)

H_β = h(β) # post_dist(x) = det(h(x))^.5 * exp(.5 * (x .- β_true)' * h(x) * (x .- β_true))

# post_dist(x, H) = pdf(MultivariateNormal(β_true, (H .+ H')./2), x)

post_dist(x, H) = pdf(MultivariateNormal(zeros(nparam), (H .+ H')./2), x .- β_true)


# Set M-H hyperparameters ...
# ---------------------------

# Set prior parameters

nparam = 5

β_lag = rand(nparam) 

nIter = 60000 # Set number of iterations...

# Distributions ...

s = 2.93/sqrt(nparam)

proposal_distribution(params, H) = MvNormal(params, H .* s^2) # Proposal distribution 

u_dist = Uniform(0,1) # Uniform distribution

# Prior parameter results

lik_lag = post_dist(β_lag, h(β_lag)); # Initialize markov state

H_inv_lag = inv(h(β_lag))

H_inv_lag = (H_inv_lag + H_inv_lag')./2

θ_accepted_list = zeros(nIter, nparam)

Accepted_flag = zeros(nIter)

jj = 0 

while jj < nIter

    jj += 1

    u = rand(u_dist)

    # Make proposal (conditioned on previous parameters), draw until get values greater than 0 for variances ...

    kk = sample(1:nparam,1)[1]
    
    β_proposed = β_lag .+ rand(proposal_distribution(zeros(nparam), H_inv_lag))

    # β_proposed = rand(proposal_distribution(β_lag, H_inv_lag))

    H_inv_prop = inv(h(β_proposed))

    H_inv_prop = (H_inv_prop + H_inv_prop')./2

    lik_proposed = post_dist(β_proposed, h(β_proposed))

    # Sample 

    pdf_β_proposed = pdf(proposal_distribution(zeros(nparam), H_inv_lag), β_proposed .- β_lag)
    
    pdf_β_lag = pdf(proposal_distribution(zeros(nparam), H_inv_prop), β_lag .- β_proposed)
    
    # Compute acceptance rate
    γ = (lik_proposed/lik_lag) * (pdf_β_lag/pdf_β_proposed)
             

    if u <= min(1, γ)# Accept

        θ_accepted_list[jj,:] = β_proposed

        β_lag = copy(β_proposed)  # Accepted proposal is laged for next period

        lik_lag = copy(lik_proposed)  # Likelihood Associated

        H_inv_lag = copy(H_inv_prop)

        Accepted_flag[jj] = 1
        
    else # Not accepted

        θ_accepted_list[jj,:] = β_lag
        
    end

end



burnIn = 500

# Plot histogram with accepted proposals for σ

plot(θ_accepted_list[burnIn:end,1], label="β0 = 1")
plot!(θ_accepted_list[burnIn:end,2], label="β0 = 2")
plot!(θ_accepted_list[burnIn:end,3], label="β0 = 3")
plot!(θ_accepted_list[burnIn:end,4], label="β0 = 4")
plot!(θ_accepted_list[burnIn:end,5], label="β0 = 5")


histogram(θ_accepted_list[burnIn:end,1])
histogram!(θ_accepted_list[burnIn:end,2])
histogram!(θ_accepted_list[burnIn:end,3])
histogram!(θ_accepted_list[burnIn:end,4])
histogram!(θ_accepted_list[burnIn:end,5])

vline!([.6,1.2], label= "True Values")

savefig("question_4_sigmas.pdf")

# mean(Accepted_flag)

#######


function prob_hybrid_ij(pr_j, j, j_lag, p, γ0, γ)

    μ = logistic(p[j_lag], γ0, γ)

    if j == j_lag
        s_ij = (1-μ) + μ * pr_j
    else
        s_ij = μ * pr_j
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
            pr_j = prob_asc_ij(Ω_i, u_i, j, p, γ0, γ)
            sj = prob_hybrid_ij(pr_j, j, j_lag, p, γ0, γ)

            # println(sj)
            if (sj>0) & (sj<1)
                logL += log(sj)/(N*(T-1))
            end

        end

    end

    return -logL

end
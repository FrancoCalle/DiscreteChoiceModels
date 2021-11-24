function logit_hybrid(θ, Y_t, XX_list, p_t, Rank_t)

    N = size(XX_list[1],1)
    K = length(XX_list)
    T = length(Y_t)

    β = θ[1:K]
    γ0 = θ[K+1]
    γ = θ[K+2]
    ϕ0 = θ[K+3]
    ϕ = θ[K+4]

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
            sj = prob_dsc_ij(pr_j, j, j_lag, p, ϕ0, ϕ)

            # println(sj)
            if (sj>0) & (sj<1)
                logL += log(sj)/(N*(T-1))
            end

        end

    end

    return -logL

end
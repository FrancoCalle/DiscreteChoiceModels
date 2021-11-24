# ASC: Alternative Specific Model:
#---------------------------------


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

function logit_asc(θ, Y, XX_list, p, Rank)

    N = size(XX_list[1],1)
    K = length(XX_list)

    β = θ[1:K]
    γ0 = θ[K+1]
    γ = θ[K+2]

    u = zeros(size(XX_list[1])); for k in 1:K u .+= XX_list[k].*β[k] end 
    
    logL = 0

    for ii in 1:N
        j = Y[ii]
        Ω_i = Rank[ii]
        u_i = u[ii,:]
        sj = prob_asc_ij(Ω_i, u_i, j, p, γ0, γ)
        if sj>0
            logL += log(sj)/N
        end
    end

    return -logL

end
logistic(p, γ0, γ) = exp(γ0 + p*γ)/(1 + exp(γ0 + p*γ))

P(Ω_i,j) = [C for C in collect(powerset(Ω_i)) if j in C]    


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

function choiceProbability_ij(C, u_i, j)

    u_j = u_i[j]
    u_C = u_i[C]
    Pr_j = exp(u_j)/sum(exp.(u_C))

    return Pr_j

end

function prob_ij(Ω_i, u_i, j, p, γ0, γ)

    Cset = P(Ω_i,j)
    π_C = considerationProbability_C.(Cset, Ref(Ω_i), j, Ref(p), γ0, γ)
    s_C = choiceProbability_ij.(Cset, Ref(u_i), j)
    s_ij = sum(π_C.*s_C)

    return s_ij

end

function logit_asc(θ, Y, XX_list, p, Rank)

    β = θ[1:5]
    γ0 = θ[6]
    γ = θ[7]

    N = size(XX_list[1],1)
    K = length(XX_list)
    u = zeros(size(XX_list[1])); for k in 1:K u .+= XX_list[k].*β[k] end 

    logL = 0

    for ii in 1:N
        j = Y[ii]
        Ω_i = Rank[ii]
        u_i = u[ii,:]
        sj = prob_ij(Ω_i, u_i, j, p, γ0, γ)
        if sj>0
            logL += log(sj)/N
        end
    end

    return -logL

end
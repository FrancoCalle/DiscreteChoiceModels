
logistic(p, γ0, γ) = exp(γ0 + p*γ)/(1 + exp(γ0 + p*γ))

P(Ω_i,j) = [C for C in collect(powerset(Ω_i)) if j in C]    

function choiceProbability_ij(C, u_i, j)

    u_j = u_i[j]
    u_C = u_i[C]
    Pr_j = exp(u_j)/sum(exp.(u_C))

    return Pr_j
end


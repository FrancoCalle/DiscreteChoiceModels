function logit(β, Y, XX_list, Ω)

    N = size(XX_list[1],1)
    K = length(XX_list)
    
    u = zeros(size(XX_list[1])); 
    
    for k in 1:K
        u .+= XX_list[k].*β[k] 
    end 

    u .= u.*Ω
    u[.~Ω] .= -Inf
    max_u = maximum.(eachrow(u))

    Pr = exp.(u .- max_u) ./sum.(eachrow(exp.(u .- max_u)))
    Pr_i = [Pr[ii,:][Int(Y[ii])] for ii in 1:N]
    
    logL = mean(log.(Pr_i[.~isnan.(Pr_i)]))
    
    return -logL

end


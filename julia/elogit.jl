function logit(β, Y, XX_list, Ω)

    N = size(XX_list[1],1)
    K = length(XX_list)
    
    u = zeros(size(XX_list[1])); for k in 1:K u .+= XX_list[k].*β[k] end 
    u .= u.*Ω

    for ii in 1:N
        
        # Get feasible set:
        ω = findall(x->x==1, Ω[1,:])
        u_i = u[ii,:]s
        u_if = u_i[ω]

        for jj in 1:size(ω)

            exp.(u .- max_u)./sum.(eachrow(exp.(u .- max_u)))

        end
    end


    u[.~Ω] .= -Inf
    max_u = maximum.(eachrow(u))

    Pr = exp.(u .- max_u)./sum.(eachrow(exp.(u .- max_u)))
    Pr_i = [Pr[ii,:][Int(Y[ii])] for ii in 1:N]
    
    logL = mean(log.(Pr_i[.~isnan.(Pr_i)]))
    
    return -logL

end
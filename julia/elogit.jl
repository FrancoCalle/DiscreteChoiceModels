function elogit(β, Rank, XX_list, Ω)

    N = size(XX_list[1],1)
    K = length(XX_list)
    
    u = zeros(size(XX_list[1])); for k in 1:K u .+= XX_list[k].*β[k] end 
    u .= u.*Ω

    logL = 0

    for ii in 1:N  

        ω = Rank[ii]
        u_if = u[ii,:][Int.(ω)]

        for jj in 1:length(ω)
            logL += log(exp.(u_if[jj])./sum(exp.(u_if[jj:end])))/N
        end

    end
    
    return -logL

end

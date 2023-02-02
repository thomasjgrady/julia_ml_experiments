function adam_update!(θ::AbstractArray{<:Number}, g, α, β1, β2, m, v, t; ϵ=1e-8, λ=0.0)
    m .*= β1
    v .*= β2
    m .+= (1-β1).*g    
    v .+= (1-β2).*g.^2
    mh = m./(1-β1^t)
    vh = v./(1-β2^t)
    θ .-= α.*(mh./(sqrt.(vh) .+ ϵ) .+ λ.*θ)
end

function adam_update!(θ::AbstractArray{<:AbstractArray}, g, α, β1, β2, m, v, t; ϵ=1e-8, λ=0.0)
    n = length(θ)
    for i in 1:n
        adam_update!(θ[i], g[i], α, β1, β2, m[i], v[i], t; ϵ=ϵ, λ=λ)
    end
end
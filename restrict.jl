using ChainRulesCore

include("common.jl")

function restrict(x::X, ranges) where {T,X<:AbstractArray{T}}
    n_out = sum(prod(map(length, r)) for r in ranges)
    y = zeros_like(x, T, n_out)
    offset = 0
    for r in ranges
        l = prod(map(length, r))
        y[offset+1:offset+l] .= vec(view(x, r...))
        offset += l
    end
    return y
end

function restrict_adj(y::Y, ranges, shape) where {T,Y<:AbstractArray{T}}
    x = zeros_like(y, T, shape...)
    offset = 0
    for r in ranges
        l = prod(map(length, r))
        vec(view(x, r...)) .+= y[offset+1:offset+l]
        offset += l
    end
    return x
end

function ChainRulesCore.rrule(::typeof(restrict), x, ranges)
    y = restrict(x, ranges)
    function ∇restrict(dy)
        dx = restrict_adj(dy, ranges, size(x))
        return NoTangent(), dx, NoTangent()
    end
    return y, ∇restrict
end

function ChainRulesCore.rrule(::typeof(restrict_adj), y, ranges, shape)
    x = restrict_adj(y, ranges, shape)
    function ∇restrict_adj(dx)
        dy = restrict(dx, ranges)
        return NoTangent(), dy, NoTangent(), NoTangent()
    end
    return x, ∇restrict_adj
end
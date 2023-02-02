using ChainRulesCore
using CUDA
using Zygote

zeros_like(::AbstractArray, T, dims...) = zeros(T, dims...)
zeros_like(::CuArray, T, dims...) = CUDA.zeros(T, dims...)

function swapdims(x, d0, d1)
    n = length(size(x))
    perm = collect(1:n)
    perm[d0], perm[d1] = perm[d1], perm[d0]
    x = permutedims(x, perm)
end

function ChainRulesCore.rrule(::typeof(swapdims), x, d0, d1)
    y = swapdims(x, d0, d1)
    function ∇swapdims(dy)
        return NoTangent(), swapdims(dy, d1, d0), NoTangent(), NoTangent()
    end
    return y, ∇swapdims
end

function as_matrix(x)
    s = size(x)
    return reshape(x, s[1], prod(s[2:end]))
end

cpu(x::AbstractArray{<:Number}) = x
cpu(x::CuArray{<:Number}) = Array(x)
cpu(x::AbstractArray{<:AbstractArray}) = cpu.(x)

macro _cpu_helper(x)
    quote
        $(Base.typename(typeof($x)).wrapper)([cpu(getfield($x, i)) for i in 1:fieldcount($x)]...)
    end
end
cpu(x) = @_cpu_helper x

gpu(x::AbstractArray{<:Number}) = CuArray(x)
gpu(x::CuArray{<:Number}) = x
gpu(x::AbstractArray{<:AbstractArray}) = gpu.(x)

macro _gpu_helper(x)
    quote
        $(Base.typename(typeof($x)).wrapper)([gpu(getfield($x, i)) for i in 1:fieldcount($x)]...)
    end
end
gpu(x) = @_gpu_helper x
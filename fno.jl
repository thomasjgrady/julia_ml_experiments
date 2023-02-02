using ChainRulesCore
using CUDA
using FFTW
using NNlib
using NNlibCUDA
using OMEinsum
using Parameters
using Zygote

include("common.jl")
include("restrict.jl")

function lift(x, Wc, Wt, bc, bt)

    s = size(x)
    n = length(s)
    c = size(Wc)[1]
    x = as_matrix(x)
    x = gelu.(Wc*x .+ bc)
    x = reshape(x, c, s[2:end]...)
    x = swapdims(x, 1, n-1)

    s = size(x)
    t = size(Wt)[1]
    x = as_matrix(x)
    x = gelu.(Wt*x .+ bt)
    x = reshape(x, t, s[2:end]...)
    return swapdims(x, 1, n-1)

end

function fno_block(x::X, D, W, ranges) where {T,X<:AbstractArray{T}}

    s = size(x)
    n = length(s)
    b = s[end]
    t = s[end-1]
    
    # Take real FFT
    # Have to permute so CUDA kernel doesn't complain about a discontiguous region
    x_ft = swapdims(x, 1, n-1)
    x_ft = rfft(x_ft, collect(1:n-1))
    x_ft = swapdims(x_ft, 1, n-1)
    s_ft = size(x_ft)

    # Restrict
    x_ft_r = restrict(x_ft, ranges)

    @show size(x_ft_r)

    # Multiply with weight
    (d, c, ns) = size(D)
    x_ft_r = reshape(x_ft_r, c, ns, b)
    y_ft_r = ein"dcn,cnb->dnb"(D, x_ft_r)

    # Un-restrict
    y_ft = restrict_adj(vec(y_ft_r), ranges, s_ft)

    # Inverse fft
    y_ft = swapdims(y_ft, 1, n-1)
    y_ft = irfft(y_ft, t, collect(1:n-1))
    y1   = swapdims(y_ft, 1, n-1)

    # Mix channel
    x  = as_matrix(x)
    y2 = W*x
    y2 = reshape(y2, s...)

    return gelu.(y1 .+ y2)

end

function proj(x, Wc, bc)
    s = size(x)
    c = size(Wc)[1]
    x = as_matrix(x)
    x = Wc*x .+ bc
    x = reshape(x, c, s[2:end]...)
    return x
end

@with_kw struct FNOConfig
    T = Float64
    device = "cpu"
    shape
    modes; @assert length(modes) == length(shape)
    batch_size = 1
    timesteps_in = 1
    input_dim = length(shape) + 1
    output_dim = 1
    lifted_dim = 20
    num_blocks = 4
end

mutable struct FNOParams{T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T},A<:AbstractArray{Complex{T},3}}
    Wc_in::M
    Wt_in::M
    bc_in::V
    bt_in::V
    Ds::Vector{A}
    Ws::Vector{M}
    ranges::Vector
    Wc_out::M
    bc_out::V

    function FNOParams(config::FNOConfig)

        n = length(config.shape)
        range_axes = [[1:config.lifted_dim]]
        for (i, m) in enumerate(config.modes)
            if i < n
                s = config.shape[i]
                push!(range_axes, [1:m, s-m+1:s])
            else
                push!(range_axes, [1:m])
            end
        end

        ranges = vec(collect(Iterators.product(range_axes...)))
        n_restrict = sum(map(rs -> prod(map(length, rs)), ranges))

        namespace = Base
        Wc_in = namespace.rand(config.T, config.lifted_dim, config.input_dim)./(config.lifted_dim*config.input_dim)
        Wt_in = namespace.rand(config.T, config.shape[end], config.timesteps_in)./(config.shape[end]*config.timesteps_in)
        bc_in = namespace.zeros(config.T, config.lifted_dim)
        bt_in = namespace.zeros(config.T, config.shape[end])
        Ds = [
            namespace.rand(
                Complex{config.T},
                config.lifted_dim,
                config.lifted_dim,
                n_restrict ÷ config.lifted_dim
            )./(config.lifted_dim*config.lifted_dim)
        for _ in 1:config.num_blocks]
        Ws = [
            namespace.rand(
                config.T,
                config.lifted_dim,
                config.lifted_dim
            )./(config.lifted_dim*config.lifted_dim)
        for _ in 1:config.num_blocks]
        Wc_out = namespace.rand(config.T, config.output_dim, config.lifted_dim)./(config.output_dim*config.lifted_dim)
        bc_out = namespace.zeros(config.T, config.output_dim)

        return new{config.T,typeof(Wc_in),typeof(bc_in),typeof(Ds[1])}(
            Wc_in,
            Wt_in,
            bc_in,
            bt_in,
            Ds,
            Ws,
            ranges,
            Wc_out,
            bc_out
        )
    end

    function FNOParams(args...)
        T = eltype(args[1])
        M = typeof(args[1])
        V = typeof(args[3])
        A = typeof(args[5][1])
        return new{T,M,V,A}(args...)
    end
end

function fno(x::X, params::FNOParams{T,M,V,A}) where {T,X<:AbstractArray{T},M,V,A}
    x = lift(x, params.Wc_in, params.Wt_in, params.bc_in, params.bc_out)
    for (D, W) in zip(params.Ds, params.Ws)
        x = fno_block(x, D, W, params.ranges)
    end
    x = proj(x, params.Wc_out, params.bc_out)
    return x
end

config = FNOConfig(shape = [64, 64, 20], modes = [8, 8, 8])
params = FNOParams(config) |> gpu;
x = rand(config.input_dim, config.shape[1:end-1]..., config.timesteps_in, config.batch_size)
y = fno(x, params) |> gpu
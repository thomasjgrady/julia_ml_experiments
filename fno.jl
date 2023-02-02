using Distributed

# Add workers
addprocs(4)

@everywhere begin

using ChainRulesCore
using CUDA
using FFTW
using LinearAlgebra
using MAT
using NNlib
using NNlibCUDA
using OMEinsum
using ParallelOperations
using Parameters
using Plots
using Printf
using ProgressBars
using Random
using Transducers
using Zygote

import Base.zero

end # @everywhere (separate block for loading macros)

@everywhere begin

include("common.jl")
include("restrict.jl")
include("adam.jl")
include("dataparallel.jl")

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

function fno_rfft(x::X) where {T,X<:AbstractArray{T}}
    n = length(size(x))
    x = swapdims(x, 1, n-1)
    x = rfft(x, collect(1:n-2))
    x = swapdims(x, 1, n-1)
    return x
end

function fno_irfft(x::X, t) where {T,X<:AbstractArray{T}}
    n = length(size(x))
    x = swapdims(x, 1, n-1)
    x = irfft(x, t, collect(1:n-2))
    x = swapdims(x, 1, n-1)
    return x
end

function fno_block(x::X, D, W, ranges) where {T,X<:AbstractArray{T}}

    s = size(x)
    n = length(s)
    b = s[end]
    t = s[end-1]
    
    # Forward RFFT
    x_ft = fno_rfft(x)
    s_ft = size(x_ft)

    # Restrict, weight multiply, adjoint_restrict
    (d, c, ns) = size(D)
    x_ft = restrict(x_ft, ranges)
    x_ft = reshape(x_ft, c, ns, b)
    y_ft = ein"dcn,cnb->dnb"(D, x_ft)
    y_ft = restrict_adj(vec(y_ft), ranges, s_ft)

    # Inverse RFFT
    y0 = fno_irfft(y_ft, t)

    # Passthrough weighting
    x  = as_matrix(x)
    y1 = W*x
    y1 = reshape(y1, s...)

    # Nonlinearity
    return gelu.(y0 .+ y1)

end

function proj(x, Wc, bc)
    s = size(x)
    c = size(Wc)[1]
    x = as_matrix(x)
    x = Wc*x .+ bc
    x = reshape(x, c, s[2:end]...)
    return x
end

function fno(x, Wc_in, Wt_in, bc_in, bt_in, Ds, Ws, Wc_out, bc_out, ranges)
    x = lift(x, Wc_in, Wt_in, bc_in, bt_in)
    for (D, W) in zip(Ds, Ws)
        x = fno_block(x, D, W, ranges)
    end
    x = proj(x, Wc_out, bc_out)
    return x
end

@with_kw struct FNOConfig
    T = Float64
    shape
    modes; @assert length(modes) == length(shape)
    batch_size = 1
    timesteps_in = 1
    input_dim = length(shape) + 1
    output_dim = 1
    lifted_dim = 20
    num_blocks = 4
end

function fno_init(config::FNOConfig)

    # Extract type information
    T = config.T

    # Compute ranges
    range_axes = [[1:config.lifted_dim]]
    for (i, m) in enumerate(config.modes)
        if i == length(config.modes)
            push!(range_axes, [1:m])
        else
            s = config.shape[i]
            push!(range_axes, [1:m, s-m+1:s])
        end
    end
    push!(range_axes, [1:config.batch_size])
    ranges = vec(collect(Iterators.product(range_axes...)))

    # Extract shape information
    d_in   = config.input_dim
    d_lift = config.lifted_dim
    d_out  = config.output_dim
    t_in   = config.timesteps_in
    t_out  = config.shape[end]
    n_res  = sum([prod(map(length, rs[1:end-1])) for rs in ranges])

    # Initialize weights
    Wc_in = rand(T, d_lift, d_in)./sqrt(d_lift*d_in)
    Wt_in = rand(T, t_out, t_in)./sqrt(t_out*t_in)
    bc_in = zeros(T, d_lift)
    bt_in = zeros(T, t_out)

    Ds = [rand(Complex{T}, d_lift, d_lift, n_res÷d_lift)./(d_lift^2) for _ in 1:config.num_blocks]
    Ws = [rand(T, d_lift, d_lift)./(d_lift^2) for _ in 1:config.num_blocks]

    Wc_out = rand(T, d_out, d_lift)./sqrt(d_out*d_lift)
    bc_out = zeros(T, d_out)

    return [Wc_in, Wt_in, bc_in, bt_in, Ds, Ws, Wc_out, bc_out], ranges
end

end # @everywhere

# ==========================
#          Training
# ==========================

# Reproducibility
Random.seed!(1337)

# Load data
data_path = joinpath(homedir(), "data/NavierStokes_V1e-5_N1200_T20.mat");
u = Float64.(matread(data_path)["u"]);

# Reshape data
subsample = 1
u = permutedims(u, (2, 3, 4, 1));
u = u[1:subsample:end,1:subsample:end,:,:];
(nx, ny, nt, nb) = size(u);
u = reshape(u, 1, nx, ny, nt, nb);

# Separate training data
train_split = 0.8;
split_idx = Int(round(train_split*nb));
n_train = split_idx
n_test  = nb - n_train

x_train = u[:,:,:,1:1,1:split_idx];
y_train = u[:,:,:,:,1:split_idx];
x_test  = u[:,:,:,1:1,split_idx+1:end];
y_test  = u[:,:,:,:,split_idx+1:end];

# Configure network
device = cpu;
config = FNOConfig(shape = [nx, ny, nt], modes = [8, 8, 8], input_dim = 1, batch_size=5);
θ, ranges = fno_init(config);

α  = 1e-3 # Learning rate
λ  = 1e-4 # Weight decay
β1 = 0.9   # First ADAM decay parameter
β2 = 0.999 # Second ADAM decay parameter
m  = [0.0.*p for p in θ]; # First moment vector for ADAM
v  = [0.0.*p for p in θ]; # Second moment vector for ADAM

# Setup optimization
num_epochs = 20;

@everywhere begin
    # Gradient reducer
    reduce_grads(g) = g
    reduce_grads(g::AbstractArray{<:Number}, h::AbstractArray{<:Number}) = g .+ h
    reduce_grads(g::AbstractArray{<:AbstractArray}, h::AbstractArray{<:AbstractArray}) = [reduce_grads(u, v) for (u, v) in zip(g, h)]
end

# Main training loop
for epoch in 1:num_epochs

    # Get batch schedule
    schedule = [i:i+config.batch_size-1 for i in 1:config.batch_size:n_train]
    Random.shuffle!(schedule)
    nbatches = length(schedule)

    # Training iterations
    pbar = ProgressBar(1:nworkers():nbatches)
    for i in pbar

        # Get training pairs
        xs = [view(x_train, :,:,:,:,schedule[j]) for j in i:min(i+nworkers()-1,nbatches)];
        ys = [view(y_train, :,:,:,:,schedule[j]) for j in i:min(i+nworkers()-1,nbatches)];

        @sca

        # Compute local losses and pullbacks
        @everywhere begin
            loss, pullback = Zygote.pullback()
        end
        losses_and_pullbacks = pmap(pullback_producer, pairs)
        loss = sum(map(first, losses_and_pullbacks))/nworkers()
        pullbacks = collect(map(last, losses_and_pullbacks))

        # Parameters are in third slot
        @everywhere apply_pullback(pb) = collect(pb($loss)[3])

        # Compute local gradients and sumreduce
        foldxd(reduce_grads, Map(apply_pullback), pullbacks)

        set_description(pbar, string(@sprintf("epoch = %03d, batch = %03d, loss = %.4f", epoch, i, loss)))
    end
    break
end

ŷ = fno(view(x_train, :,:,:,:,1:1) |> device, θ..., ranges) |> cpu;
clim = (minimum(ŷ)-0.05, maximum(ŷ)+0.05)
anim = @animate for i in 1:nt
    heatmap(ŷ[1,:,:,i,1]; clim=clim)
end;
gif(anim, "out.gif", fps=5);
using Distributed
using Zygote

"""
Given a list of training pairs on a master worker, performs a forward/adjoint pass
through f in a data-parallel fashion. f is assumed to be a function (f, x, args...).
Returns the grads which is the gradient w.r.t x, y and the arguments to f
"""
function distributed_training_iteration(f, training_pairs, args...)
    @everywhere get_pullback(data_pair) = Zygote.pullback(f, data_pair..., args...)
    losses_and_pullbacks = pmap(get_pullback, training_pairs)
    loss = sum(map(first, losses_and_pullbacks))/nworkers()
    pullbacks = collect(map(last, losses_and_pullbacks))
    @everywhere begin
        grads = $pullbacks[myid()]($loss)
        return $loss, grads
    end
end
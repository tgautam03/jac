mutable struct Tensor
    val::Union{Number, Array}       # Value of Tensor
    creators::Vector                # Parents of this Tensor
    grads::Vector                   # Gradient Functions wrt creators

    # Constructor 1
    function Tensor(val::Array)
        new(val, [], [])
    end

    # Constructor 2
    function Tensor(val::Union{Number,Array}, creators::Vector{Tensor}, grads::Vector)
        @assert (size(creators)[1] == size(grads)[1]) # Make sure grads for each creator is provided
        new(val, creators, grads)
    end
end
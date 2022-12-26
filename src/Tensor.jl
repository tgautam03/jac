mutable struct Tensor
    val::Array          # Value of Tensor
    creators::Vector    # Parents of this Tensor
    local_grads::Vector # Local Gradient wrt creators

    # Constructor 1
    function Tensor(val::Array)
        new(val, [], [])
    end

    # Constructor 2
    function Tensor(val::Array, creators::Vector{Tensor}, local_grads::Vector{<:Array})
        @assert (size(creators)[1] == size(local_grads)[1]) # Make sure local_grads for each creator is provided
        new(val, creators, local_grads)
    end
end
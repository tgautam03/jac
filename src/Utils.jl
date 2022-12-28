using Base

include("Tensor.jl")

# Printing Tensor Nicely
function Base.show(io::IO, T::Tensor)
    print(io, "Value = $(T.val)\nLocal Gradients = $(T.grads)")
end

# size function on Tensor
function Base.size(T::Tensor)
    return size(T.val)
end
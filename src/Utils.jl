using Base

include("Tensor.jl")

# Printing Tensor Nicely
function Base.print(T::Tensor)
    print(T.val)
end

function Base.println(T::Tensor)
    println(T.val)
end

# size function on Tensor
function Base.size(T::Tensor)
    return size(T.val)
end
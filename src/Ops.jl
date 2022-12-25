using Base

include("Tensor.jl")

# Addition Operation on Tensor types
function Base.:+(T1::Tensor, T2::Tensor)
    return T1.val + T2.val
end
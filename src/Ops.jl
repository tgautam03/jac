using Base

include("Tensor.jl")

# Addition Operation on Tensor types
function Base.:+(T1::Tensor, T2::Tensor)
    return Tensor(T1.val + T2.val, [T1, T2], [ones(size(T1)), ones(size(T2))])
end

# Multiplication Operation on Tensor types
function Base.:*(T1::Tensor, T2::Tensor)
    return Tensor(T1.val * T2.val, [T1, T2], [T2.val, T1.val])
end
module jac

include("Tensor.jl")
export Tensor

include("Ops/Addition.jl")
include("Ops/Multiplication.jl")

include("AutoGrad.jl")


end # module JAC

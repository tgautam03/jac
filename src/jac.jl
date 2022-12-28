module jac

# Including and Exporting Tensor Data Type
include("Tensor.jl")
export Tensor

# Including and Exporting Operations
include("Ops.jl")
export *
export +
export -
export sum
export ^
export /

# Including and Exporting Utilities functions
include("Utils.jl")
export show
export size

# Including autograd
include("AutoGrad.jl")

end # module jac

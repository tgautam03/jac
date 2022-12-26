module jac

# Including and Exporting Tensor Data Type
include("Tensor.jl")
export Tensor

# Including and Exporting Operations
include("Ops.jl")
export +

# Including and Exporting Utilities functions
include("Utils.jl")
export print
export println

# Including autograd
include("autograd.jl")
export grad

end # module jac

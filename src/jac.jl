module jac

using Base

include("Variable.jl")
export Variable

include("autograd.jl")
export autograd

include("ops/add.jl")
export +

include("ops/mul.jl")
export *

end # module jac

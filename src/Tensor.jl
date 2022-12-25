mutable struct Tensor
    val::Array # Value of Tensor

    # Constructor
    function Tensor(val::Array)
        new(val)
    end
end
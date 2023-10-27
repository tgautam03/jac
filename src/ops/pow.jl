function Base.:^(var1::Variable, pow::Int)
    # Performing power
    value = var1.value^pow

    # Local Gradients Computations
    global_dvar1 = global_grad::Variable -> global_grad * pow * var1^(pow-1)

    return Variable(value, [var1], [global_dvar1])
end
function Base.:sin(var1::Variable)
    # Performing Addition
    value = sin(var1.value)

    # Local Gradients Computations
    global_dvar1 = global_grad::Variable -> global_grad * cos(var1)

    return Variable(value, [var1], [global_dvar1])
end
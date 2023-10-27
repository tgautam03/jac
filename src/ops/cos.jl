function Base.:cos(var1::Variable)
    # Performing Addition
    value = cos(var1.value)

    # Local Gradients Computations
    global_dvar1 = global_grad::Variable -> global_grad * -1 * sin(var1)

    return Variable(value, [var1], [global_dvar1])
end
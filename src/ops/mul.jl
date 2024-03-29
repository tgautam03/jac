# Multiplication Operaton on Scalars
function Base.:*(var1::Variable, var2::Variable)
    # Performing Multiplication
    value = var1.value * var2.value

    # Local Gradients Computations
    global_dvar1 = global_grad::Variable -> global_grad * var2
    global_dvar2 = global_grad::Variable -> global_grad * var1

    return Variable(value, [var1, var2], [global_dvar1, global_dvar2])
end

# Multiplication Operaton on Scalars
function Base.:*(var1::Variable, var2::Number)
    # Performing Multiplication
    value = var1.value * var2

    # Local Gradients Computations
    global_dvar1 = global_grad::Variable -> global_grad * var2

    return Variable(value, [var1], [global_dvar1])
end

# Multiplication Operaton on Scalars
function Base.:*(var1::Number, var2::Variable)
    # Performing Multiplication
    value = var1 * var2.value

    # Local Gradients Computations
    global_dvar2 = global_grad::Variable -> global_grad * var1

    return Variable(value, [var2], [global_dvar2])
end
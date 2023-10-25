# Addition Operaton on Scalars
function Base.:+(var1::Variable, var2::Variable)
    # Performing Addition
    value = var1.value + var2.value

    # Local Gradients Computations
    global_dvar1 = global_grad::Variable -> global_grad * Variable(1)
    global_dvar2 = global_grad::Variable -> global_grad * Variable(1)

    return Variable(value, [var1, var2], [global_dvar1, global_dvar2])
end
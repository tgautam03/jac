using jac

# Tensor 1
T1 = Tensor(rand(1,1))
println("Tensor 1: ", T1)

# Tensor 2
T2 = Tensor(rand(1,1))
println("Tensor 1: ", T2)

# Tensor Addition
T_sum = T1 + T2
println("T1 + T2 = ", T_sum)

# Tensor Multiplication
T_mul = T1 * T_sum
println("T1 * T_sum = ", T_mul)

# Gradients
dT_mul_dT1 = grad(T_mul)[T1]
println("Gradient of T_mul wrt T1: ", dT_mul_dT1)

dT_mul_dT2 = grad(T_mul)[T2]
println("Gradient of T_mul wrt T2: ", dT_mul_dT2)
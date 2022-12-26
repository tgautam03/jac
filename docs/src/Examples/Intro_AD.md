# Introduction to Automatic Differentiation using JAC

```julia
using jac

# Tensor 1
T1 = Tensor(rand(1,1))
println("Tensor 1: ", T1)

# Tensor 2
T2 = Tensor(rand(1,1))
println("Tensor 2: ", T2)

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
```
```sh
Tensor 1: Tensor([0.30216931737054675;;], Any[], Any[])

Tensor 2: Tensor([0.3166077631591131;;], Any[], Any[])

T1 + T2 = Tensor([0.6187770805296599;;], Tensor[Tensor([0.30216931737054675;;], Any[], Any[]), Tensor([0.3166077631591131;;], Any[], Any[])], [[1.0;;], [1.0;;]])

T1 * T_sum = Tensor([0.18697544802818716;;], Tensor[Tensor([0.30216931737054675;;], Any[], Any[]), Tensor([0.6187770805296599;;], Tensor[Tensor([0.30216931737054675;;], Any[], Any[]), Tensor([0.3166077631591131;;], Any[], Any[])], [[1.0;;], [1.0;;]])], [[0.6187770805296599;;], [0.30216931737054675;;]])

Gradient of T_mul wrt T1: [0.9209463979002066;;]

Gradient of T_mul wrt T2: [0.30216931737054675;;]
```
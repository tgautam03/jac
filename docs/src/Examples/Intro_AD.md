# Introduction to Automatic Differentiation using JAC

```julia
using jac

# Tensor 1
T1 = Tensor(rand(5,1))
println("Tensor 1: ", T1)

# Tensor 2
T2 = Tensor(rand(5,1))
println("Tensor 1: ", T2)

# Tensor Addition
T_sum = T1 + T2
println("T1 + T2 = ", T_sum)
```
```sh
Tensor 1: Tensor([0.0644575049947973; 0.2975948501433827; 0.21703058227779992; 0.30486266258723094; 0.23805453112545705;;])

Tensor 2: Tensor([0.546849262288914; 0.4356577492007686; 0.8822673779672496; 0.1786712496167836; 0.9722866473914338;;])

T1 + T2 = [0.6113067672837112; 0.7332525993441513; 1.0992979602450497; 0.48353391220401454; 1.2103411785168907;;]
```
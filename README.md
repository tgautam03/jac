# jac

So far, only scalars with very few arithmatic operations supported. See an example below.

```julia
# Importing the Library
using jac
```

```julia
# Define Function
function f(a, b, c)
    return (sin(a)*(b^3)+cos(c))*a
end
```

```julia
# Input Variables
a = Variable(4.8)
b = Variable(5.0)
c = Variable(6.5)

# Evaluating Function
v = f(a, b, c)

println("v = ", v.value)
```
```
v = -593.0111446980098
```

```julia
# Eval Derivatives
dv = autograd(v)

dv_da = dv[a]
dv_db = dv[b]
dv_dc = dv[c]

println("dv_da = ", dv_da.value)
println("dv_db = ", dv_db.value)
println("dv_dc = ", dv_dc.value)
```
```
dv_da = -71.04459841508421
dv_db = -358.61925918090265
dv_dc = -1.0325759428215144
```

```julia
# Higher Order Derivatives
d2v_da2 = autograd(dv_da)[a]
println("d2v_da2 = ", d2v_da2.value)

d3v_da3 = autograd(d2v_da2)[a]
println("d3v_da3 = ", d3v_da3.value)

d4v_da4 = autograd(d3v_da3)[a]
println("d4v_da4 = ", d4v_da4.value)
```
```
d2v_da2 = 619.573511161366
d3v_da3 = 321.06233824977244
d4v_da4 = -641.4482570212276
```
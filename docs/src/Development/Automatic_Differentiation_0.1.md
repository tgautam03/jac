# Automatic Differentiation 0.1

## Topics Covered

Following topics are covered in this post:

- Creating custom `Tensor` type.

- Addition operation on `Tensor` type.

  

## Tensor

I'll start by defining a custom `Tensor` type which will handle the data and around this I'll create all the automatic differentiation machinery. 

```julia
mutable struct Tensor
    val::Array # Value of Tensor

    # Constructor
    function Tensor(val::Array)
        new(val)
    end
end
```

`Tensor` is a `mutable struct` type which holds Julia's inbuild `Array` type (for now I'm only concerned about computations on CPU but I'll add GPU support later). A **constructor** is defined which accepts only `Array` type and initialises `val` with it.



### Addition Operation

Adding addition support to `Tensor`s is very easy, thanks to **multiple dispatch**.

```julia
function Base.:+(T1::Tensor, T2::Tensor)
    return T1.val + T2.val
end
```

Here, I've just added support for `Tensor` on the base `+` operation. 


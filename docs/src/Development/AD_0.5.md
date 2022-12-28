# Automatic Differentiation for Simple Neural Netowrks

## Topics Covered

Following topics are covered in this post:

- Adapting Automatic Differentiation (AD) for several Matrix Operations like:
    - Matrix-Matrix Multiplication
    - Matrix Addition (Including support for broadcasting)
    - Summation and Power oprations

## Expanding Operations

It's very easy to expand operations to support matrices as it only involves understanding of chain rule involving matrices.

### Addition and Subtraction
```julia
# Addition Operation on Tensor types
function Base.:+(T1::Tensor, T2::Tensor)
    if size(T1) == size(T2) 
        val = T1.val + T2.val
        creators = [T1, T2]
        grads = [path_val -> path_val .* ones(size(T1)), path_val -> path_val .* ones(size(T2))]

        return Tensor(val, creators, grads)

    elseif (size(T1)[1] == size(T2)[1] && size(T1)[2] == 1) # Broadcasting along columns when T1 is Vector
        val = T1.val .+ T2.val
        creators = [T1, T2]
        grads = [path_val -> sum(path_val .* ones(size(T1)), dims=2), path_val -> path_val .* ones(size(T2))]

        return Tensor(val, creators, grads)

    elseif (size(T1)[1] == size(T2)[1] && size(T2)[2] == 1) # Broadcasting along columns when T2 is Vector
        val = T1.val .+ T2.val
        creators = [T1, T2]
        grads = [path_val -> path_val .* ones(size(T1)), path_val -> sum(path_val .* ones(size(T2)), dims=2)]

        return Tensor(val, creators, grads)

    elseif (size(T1)[2] == size(T2)[2] && size(T1)[1] == 1) # Broadcasting along rows when T1 is Vector
        val = T1.val .+ T2.val
        creators = [T1, T2]
        grads = [path_val -> sum(path_val .* ones(size(T1)), dims=1), path_val -> path_val .* ones(size(T2))]

        return Tensor(val, creators, grads)

    elseif (size(T1)[2] == size(T2)[2] && size(T2)[1] == 1) # Broadcasting along rows when T2 is Vector
        val = T1.val .+ T2.val
        creators = [T1, T2]
        grads = [path_val -> path_val .* ones(size(T1)), path_val -> sum(path_val .* ones(size(T2)), dims=1)]

        return Tensor(val, creators, grads)

    else
        throw(DomainError(ch, "Tensor dims mismatch"))
    end
end
```
```julia
# Subtraction Operation on Tensor types
function Base.:-(T1::Tensor, T2::Tensor)
    if size(T1) == size(T2)
        val = T1.val - T2.val
        creators = [T1, T2]
        grads = [path_val -> path_val .* ones(size(T1)), path_val -> -1 .* path_val .* ones(size(T2))]

        return Tensor(val, creators, grads)

    elseif (size(T1)[1] == size(T2)[1] && size(T1)[2] == 1) # Broadcasting along columns when T1 is Vector
        val = T1.val .- T2.val
        creators = [T1, T2]
        grads = [path_val -> sum(path_val .* ones(size(T1)), dims=2), path_val -> -1 .* path_val .* ones(size(T2))]

        return Tensor(val, creators, grads)

    elseif (size(T1)[1] == size(T2)[1] && size(T2)[2] == 1) # Broadcasting along columns when T2 is Vector
        val = T1.val .- T2.val
        creators = [T1, T2]
        grads = [path_val -> path_val .* ones(size(T1)), path_val -> sum(-1 .* path_val .* ones(size(T2)), dims=2)]

        return Tensor(val, creators, grads)

    elseif (size(T1)[2] == size(T2)[2] && size(T1)[1] == 1) # Broadcasting along rows when T1 is Vector
        val = T1.val .- T2.val
        creators = [T1, T2]
        grads = [path_val -> sum(path_val .* ones(size(T1)), dims=1), path_val -> -1 .* path_val .* ones(size(T2))]

        return Tensor(val, creators, grads)

    elseif (size(T1)[2] == size(T2)[2] && size(T2)[1] == 1) # Broadcasting along rows when T2 is Vector
        val = T1.val .- T2.val
        creators = [T1, T2]
        grads = [path_val -> path_val .* ones(size(T1)), path_val -> sum(-1 .* path_val .* ones(size(T2)), dims=1)]

        return Tensor(val, creators, grads)

    else
        throw(DomainError(ch, "Tensor dims mismatch"))
    end
end
```

### Matrix-Matrix Multiplication
```julia
# Multiplication Operation on Tensor types
function Base.:*(T1::Tensor, T2::Tensor)
    return Tensor(T1.val * T2.val, [T1, T2], [path_val -> path_val * transpose(T2.val), path_val -> transpose(T1.val) * path_val])
end
```

### Sum, Power and Division Operations
```julia
# Sum function
function Base.:sum(T::Tensor)
    return Tensor(sum(T.val), [T], [path_val -> path_val .* ones(size(T))])
end
```
```julia
# Power Operation
function Base.:^(T::Tensor, power::Int)
    val = T.val .^ power
    creators = [T]
    grads = [path_val -> path_val .* power .* (T.val .^ (power-1))]

    return Tensor(val, creators, grads)
end
```
```julia
# Division by Scalar
function Base.:/(T::Tensor, divisor::Real)
    val = T.val ./ divisor
    creators = [T]
    grads = [path_val -> path_val ./ divisor]

    return Tensor(val, creators, grads)
end
```

Amazingly, we're all set. AD written this way enables modularity such that I can add any operations (along with the chain rule formula) and everything will work automatically. 

> For more details regarding usage check out [this post](https://tgautam03.github.io/jac/dev/Examples/Basic_NN/).

## References
- This work is completely inspired (and I learnt mostly) from this excellent [blog post](https://sidsite.com/posts/autodiff/) by Sidney Radcliffe who also maintains [Small Pebbel](https://github.com/sradc/SmallPebble).
- Trask, Andrew W. Grokking deep learning. Simon and Schuster, 2019.
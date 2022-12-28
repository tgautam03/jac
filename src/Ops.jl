using Base

include("Tensor.jl")

# Addition Operation on Tensor types
function Base.:+(T1::Tensor, T2::Tensor)
    if size(T1) == size(T2)
        val = T1.val + T2.val
        creators = [T1, T2]
        grads = [path_val -> path_val .* ones(size(T1)), path_val -> path_val .* ones(size(T2))]

        return Tensor(val, creators, grads)

    elseif (size(T1)[1] == size(T2)[1] && size(T1)[2] == 1)
        val = T1.val .+ T2.val
        creators = [T1, T2]
        grads = [path_val -> sum(path_val .* ones(size(T1)), dims=2), path_val -> path_val .* ones(size(T2))]

        return Tensor(val, creators, grads)

    elseif (size(T1)[1] == size(T2)[1] && size(T2)[2] == 1)
        val = T1.val .+ T2.val
        creators = [T1, T2]
        grads = [path_val -> path_val .* ones(size(T1)), path_val -> sum(path_val .* ones(size(T2)), dims=2)]

        return Tensor(val, creators, grads)

    elseif (size(T1)[2] == size(T2)[2] && size(T1)[1] == 1)
        val = T1.val .+ T2.val
        creators = [T1, T2]
        grads = [path_val -> sum(path_val .* ones(size(T1)), dims=1), path_val -> path_val .* ones(size(T2))]

        return Tensor(val, creators, grads)

    elseif (size(T1)[2] == size(T2)[2] && size(T2)[1] == 1)
        val = T1.val .+ T2.val
        creators = [T1, T2]
        grads = [path_val -> path_val .* ones(size(T1)), path_val -> sum(path_val .* ones(size(T2)), dims=1)]

        return Tensor(val, creators, grads)

    else
        throw(DomainError(ch, "Tensor dims mismatch"))
    end
end

# Subtraction Operation on Tensor types
function Base.:-(T1::Tensor, T2::Tensor)
    if size(T1) == size(T2)
        val = T1.val - T2.val
        creators = [T1, T2]
        grads = [path_val -> path_val .* ones(size(T1)), path_val -> -1 .* path_val .* ones(size(T2))]

        return Tensor(val, creators, grads)

    elseif (size(T1)[1] == size(T2)[1] && size(T1)[2] == 1)
        val = T1.val .- T2.val
        creators = [T1, T2]
        grads = [path_val -> sum(path_val .* ones(size(T1)), dims=2), path_val -> -1 .* path_val .* ones(size(T2))]

        return Tensor(val, creators, grads)

    elseif (size(T1)[1] == size(T2)[1] && size(T2)[2] == 1)
        val = T1.val .- T2.val
        creators = [T1, T2]
        grads = [path_val -> path_val .* ones(size(T1)), path_val -> sum(-1 .* path_val .* ones(size(T2)), dims=2)]

        return Tensor(val, creators, grads)

    elseif (size(T1)[2] == size(T2)[2] && size(T1)[1] == 1)
        val = T1.val .- T2.val
        creators = [T1, T2]
        grads = [path_val -> sum(path_val .* ones(size(T1)), dims=1), path_val -> -1 .* path_val .* ones(size(T2))]

        return Tensor(val, creators, grads)

    elseif (size(T1)[2] == size(T2)[2] && size(T2)[1] == 1)
        val = T1.val .- T2.val
        creators = [T1, T2]
        grads = [path_val -> path_val .* ones(size(T1)), path_val -> sum(-1 .* path_val .* ones(size(T2)), dims=1)]

        return Tensor(val, creators, grads)

    else
        throw(DomainError(ch, "Tensor dims mismatch"))
    end
end

# Multiplication Operation on Tensor types
function Base.:*(T1::Tensor, T2::Tensor)
    return Tensor(T1.val * T2.val, [T1, T2], [path_val -> path_val * transpose(T2.val), path_val -> transpose(T1.val) * path_val])
end

# Sum function
function Base.:sum(T::Tensor)
    return Tensor(sum(T.val), [T], [path_val -> path_val .* ones(size(T))])
end

# Power Operation
function Base.:^(T::Tensor, power::Int)
    val = T.val .^ power
    creators = [T]
    grads = [path_val -> path_val .* power .* (T.val .^ (power-1))]

    return Tensor(val, creators, grads)
end

# Division by Scalar
function Base.:/(T::Tensor, divisor::Real)
    val = T.val ./ divisor
    creators = [T]
    grads = [path_val -> path_val ./ divisor]

    return Tensor(val, creators, grads)
end
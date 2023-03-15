function Base.:*(T1::Tensor, T2::Tensor)
    val = T1.val * T2.val
    parents = [T1, T2]
    
    # Gradient Functions
    dT1 = path_val -> path_val * T2
    dT2 = path_val -> path_val * T1
    grad_fns = [dT1, dT2]
    
    return Tensor(val, parents, grad_fns)
end

# ╔═╡ 52cb6b88-e335-4cda-bcee-525547704ca2
function Base.:*(n::Number, T2::Tensor)
    val = n * T2.val
    parents = [T2]
    
    # Gradient Functions
    dT2 = path_val -> path_val * n
    grad_fns = [dT2]
    
    return Tensor(val, parents, grad_fns)
end

# ╔═╡ aec69e8c-5c75-46b4-a994-a9a6591d99a8
function Base.:*(T1::Tensor, n::Number)
    val = T1.val * n
    parents = [T1]
    
    # Gradient Functions
    dT1 = path_val -> path_val * n
    grad_fns = [dT1]
    
    return Tensor(val, parents, grad_fns)
end
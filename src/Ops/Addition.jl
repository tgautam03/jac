function Base.:+(T1::Tensor, T2::Tensor)
	val = T1.val + T2.val
	parents = [T1, T2]

	# Grad Functions
	dT1 = path_val -> path_val * 1
	dT2 = path_val -> path_val * 1
	grad_fns = [dT1, dT2]

	return Tensor(val, parents, grad_fns)
end

# ╔═╡ f5988491-f53e-403c-b479-6af9dbb9e1f8
function Base.:+(n::Number, T2::Tensor)
    val = n + T2.val
    parents = [T2]
    
    # Gradient Functions
    dT2 = path_val -> path_val * 1
    grad_fns = [dT2]
    
    return Tensor(val, parents, grad_fns)
end

# ╔═╡ 476bf938-b046-4edb-a9ef-ab6fc86efd54
function Base.:+(T1::Tensor, n::Number)
    val = T1.val + n
    parents = [T1]
    
    # Gradient Functions
    dT1 = path_val -> path_val * 1
    grad_fns = [dT1]
    
    return Tensor(val, parents, grad_fns)
end
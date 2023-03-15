function Base.:^(T::Tensor, n::Number)
	val = T.val^n
	parents = [T]

	# Grad Functions
	dT = path_val -> path_val * n*T^(n-1)
	grad_fns = [dT]

	return Tensor(val, parents, grad_fns)
end
function autograd(T::Tensor)
	gradients = Dict() # Dict to hold grads of T wrt all creators

	function compute(node::Tensor, path_val::Tensor)
		for (parent, grad_fn) in zip(node.parents, node.grad_fns)
			# Chain Rule
			parent_path_val = grad_fn(path_val)

			# Checking if grad present or not
			if haskey(gradients, parent)
				gradients[parent] += parent_path_val
			else
				gradients[parent] = parent_path_val
			end

			# Recusive call
			compute(parent, parent_path_val)
		end
	end

	# Output Node
	dT_dT = Tensor(1)
	compute(T, dT_dT)

	return gradients
end
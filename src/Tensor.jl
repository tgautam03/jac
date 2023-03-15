mutable struct Tensor
	val::Number 		# Value of the Tensor
	parents::Vector 	# Parents of the Tensor
	grad_fns::Vector 	# Gradient functions wrt parents

	# Constructor to initialise the Tensor
	function Tensor(val::Number)
		new(val, [], [])
	end

	# Constructor to create Tensor using operations
	function Tensor(val::Number, parents::Vector{Tensor}, grad_fns::Vector)
		@assert size(parents) == size(grad_fns)
		new(val, parents, grad_fns)
	end
end
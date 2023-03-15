# Reverse Mode Automatic Differentiation (Scalar support only)

**Development of the code in this article with proper explaination can be seen in [this YouTube video](https://www.youtube.com/watch?v=VT7yjy2Lb8k).**

## `Tensor` type
I start by creating a custom `Tensor` type which will hold the variables and the structure of the computational graph. Code can be seen in [Tensor.jl file](https://github.com/tgautam03/jac/blob/master/src/Tensor.jl).
```julia
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
```

## Operations
Next, I defined the basic operations on the `Tensor` type. Code can be seen in [Ops folder](https://github.com/tgautam03/jac/tree/master/src/Ops).

### Addition
```julia
function Base.:+(T1::Tensor, T2::Tensor)
	val = T1.val + T2.val
	parents = [T1, T2]

	# Grad Functions
	dT1 = path_val -> path_val * 1
	dT2 = path_val -> path_val * 1
	grad_fns = [dT1, dT2]

	return Tensor(val, parents, grad_fns)
end


function Base.:+(n::Number, T2::Tensor)
    val = n + T2.val
    parents = [T2]
    
    # Gradient Functions
    dT2 = path_val -> path_val * 1
    grad_fns = [dT2]
    
    return Tensor(val, parents, grad_fns)
end


function Base.:+(T1::Tensor, n::Number)
    val = T1.val + n
    parents = [T1]
    
    # Gradient Functions
    dT1 = path_val -> path_val * 1
    grad_fns = [dT1]
    
    return Tensor(val, parents, grad_fns)
end
```

### Multiplication
```julia
function Base.:*(T1::Tensor, T2::Tensor)
    val = T1.val * T2.val
    parents = [T1, T2]
    
    # Gradient Functions
    dT1 = path_val -> path_val * T2
    dT2 = path_val -> path_val * T1
    grad_fns = [dT1, dT2]
    
    return Tensor(val, parents, grad_fns)
end


function Base.:*(n::Number, T2::Tensor)
    val = n * T2.val
    parents = [T2]
    
    # Gradient Functions
    dT2 = path_val -> path_val * n
    grad_fns = [dT2]
    
    return Tensor(val, parents, grad_fns)
end


function Base.:*(T1::Tensor, n::Number)
    val = T1.val * n
    parents = [T1]
    
    # Gradient Functions
    dT1 = path_val -> path_val * n
    grad_fns = [dT1]
    
    return Tensor(val, parents, grad_fns)
end
```

### Power
```julia
function Base.:^(T::Tensor, n::Number)
	val = T.val^n
	parents = [T]

	# Grad Functions
	dT = path_val -> path_val * n*T^(n-1)
	grad_fns = [dT]

	return Tensor(val, parents, grad_fns)
end
```

## AutoGrad
Now, let's define the `autograd` engine, that'll move through the graph evaluating gradients. Code can be seen in [AutoGrad.jl file](https://github.com/tgautam03/jac/blob/master/src/AutoGrad.jl).
```julia
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
```

## Conclusion
Code is now complete to perform Automatic Differentiation on Scalars. Example Usage can be seen [here](https://tgautam03.github.io/jac/dev/Examples/Simple_Scalar_Arithmetics/).
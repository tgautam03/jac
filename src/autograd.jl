# Uses compute() to get gradients and store them in a dict
function autograd(T::Variable)
	gradients = Dict() # Dict to hold grads of T wrt all creators

    # Computes the global gradient WRT node
    function compute(node::Variable, global_grad::Variable)
        for (parent, chain_rule) in zip(node.parents, node.chain_rules)
            # Chain Rule
            new_global_grad = chain_rule(global_grad)

            # Checking if grad present or not
            if haskey(gradients, parent)
                gradients[parent] += new_global_grad
            else
                gradients[parent] = new_global_grad
            end

            # Recusive call
            compute(parent, new_global_grad)
        end
    end
    
	# Output Node
	dT_dT = Variable(1)
	compute(T, dT_dT)

	return gradients
end
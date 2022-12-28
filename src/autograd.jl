include("Tensor.jl")

function grad(T1::Tensor)
    gradients = Dict() # Dictionary to hold gradients wrt T1

    function compute_gradients(T::Tensor, path_value::Union{Real,Array})
        #=
        Description: # Function to fill up the gradients Dictionary initialised above
        T: Gradient of Tensor T
        path_value: The gradient value coming from T's children
        =#
        for (creator, grad_function) in zip(T.creators, T.grads)
            # Applying Chain Rule
            creator_path_value = grad_function(path_value)

            if haskey(gradients, creator)
                # Add the gradients where paths merge
                gradients[creator] += creator_path_value
            else
                # Store the gradient if no merging
                gradients[creator] = creator_path_value
            end

            # Recursively call this function
            compute_gradients(creator, creator_path_value)
        end
    end

    # The top level case
    compute_gradients(T1, ones(size(T1)))

    return gradients
end 
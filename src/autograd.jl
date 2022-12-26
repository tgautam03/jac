include("Tensor.jl")

function grad(T1::Tensor)
    gradients = Dict() # Dictionary to hold gradients wrt T1

    function compute_gradients(T::Tensor, path_value::Array)
        #=
        Description: # Function to fill up the gradients Dictionary initialised above
        T: Gradient of Tensor T
        path_value: The gradient value coming from T's children
        =#
        for (creator, local_grad) in zip(T.creators, T.local_grads)
            # Applying Chain Rule
            creator_path_value = path_value .* local_grad

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
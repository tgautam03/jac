# Automatic Differentiation v0.1

## Topics Covered

Following topics are covered in this post:

- Automatic Differentiation (AD) framework.

- Creating custom `Tensor` type.

- Several operations on `Tensor` type:
    - Addition
    - Multiplication

- Automatic Differentiation using `Tensor`s.

## Framework

The framework for AD is all about graph. Let's look at how this works with a simple example:

$$A = 
\begin{bmatrix}
5 
\end{bmatrix}$$

$$B = 
\begin{bmatrix}
3 
\end{bmatrix}$$

$$C = A + B = 8 \\
D = C * B = 24$$

The above defined computations can be written as a graph shown in Figure 1.

<div class="imgcap">
<img src="https://raw.githubusercontent.com/tgautam03/jac/master/docs/src/img/Forward.png">
<div class="thecap">Figure 1: Computations as a graph </div>
</div>

Now, if we want to compute say $\frac{\partial D}{\partial B}$, we just backtrack on the paths from $D$ to $B$ and add up the values of the paths. Let's look at this again:
- We can see that $D$ results from $C$ and $B$
    - **The path along $B$**: 
        - We can write **local gradient** of $D$ with respect to $B$ as equal to $C$, i.e. $\hat{\frac{\partial D}{\partial B}}=C=8$
    - **The path along $C$**: 
        - We can write **local gradient** of $D$ with respect to $C$ as equal to $B$, i.e. $\hat{\frac{\partial D}{\partial C}}=B=3$
        - For **The path along $B$ through $C$**, we can write **local gradient** of $C$ with respect to $B$ as equal to $1$, i.e. $\hat{\frac{\partial C}{\partial B}}=1$.
        - Using chain rule **along this path**, we can write $\hat{\frac{\partial D}{\partial B}}=\hat{\frac{\partial D}{\partial C}}*\hat{\frac{\partial C}{\partial B}}=B*1=3$.
- Now, we have $\hat{\frac{\partial D}{\partial B}}$ from two different paths which merge together, hence to get final answer we just add them up and get $11$.

<div class="imgcap">
<img src="https://raw.githubusercontent.com/tgautam03/jac/master/docs/src/img/Backward.png">
<div class="thecap">Figure 1: Gradient Computations through a graph </div>
</div>

Let's now dig into the coding details.

## Tensor

I'll start by defining a custom `Tensor` type which will handle the data and around this I'll create all the automatic differentiation (AD) machinery. 

```julia
mutable struct Tensor
    val::Array          # Value of Tensor
    creators::Vector    # Parents of this Tensor
    local_grads::Vector # Local Gradient wrt creators

    # Constructor 1
    function Tensor(val::Array)
        new(val, [], [])
    end
end
```

`Tensor` is a `mutable struct` type which holds:
- Julia's inbuild `Array` type (for now I'm only concerned about computations on CPU but I'll add GPU support later) as the `val`ue of the `Tensor`.
- `creators` store the `Tensor`s used to create this `Tensor`, i.e. the parents.
- `local_grads` contain the gradient of `Tensor` with respect to each of it's parents.  

A **constructor** is defined which accepts only `Array` type and initialises `val` with it (rest is kept empty). This will define the **leaf nodes** in the graph. 

Another constructor is defined that will mainly be used to create intermediate `Tensor`s by several operations (like addition or multiplication). 

### Operations on Tensors

Adding addition and multiplication support to `Tensor`s is very easy, thanks to **multiple dispatch**.

```julia
# Addition Operation on Tensor types
function Base.:+(T1::Tensor, T2::Tensor)
    return Tensor(T1.val + T2.val, [T1, T2], [ones(size(T1)), ones(size(T2))])
end

# Multiplication Operation on Tensor types
function Base.:*(T1::Tensor, T2::Tensor)
    return Tensor(T1.val * T2.val, [T1, T2], [T2.val, T1.val])
end
```

> I've defined several utility functions like `println`, `print`, `size`, etc for `Tensor` type too.

## Automatic Differentiation 

Coding AD is relatively simple (if you understand chain rule and recursive programming) and the code with proper comments is shown below.

```julia
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
```

Let's now run the code:
```julia
A = Tensor([5][:,:]) # My AD only supports Arrays
B = Tensor([3][:,:]) # My AD only supports Arrays

C = A + B 
D = B * C

dD_dA = grad(D)[A] # dD_dA = 3
dD_dB = grad(D)[B] # dD_dB = 11
dD_dC = grad(D)[C] # dD_dC = 3
```

## References
- This work is completely inspired (and I learnt mostly) from this excellent [blog post](https://sidsite.com/posts/autodiff/) by Sidney Radcliffe who also maintains [Small Pebbel](https://github.com/sradc/SmallPebble)
- Trask, Andrew W. Grokking deep learning. Simon and Schuster, 2019.
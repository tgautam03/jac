# Introduction to Reverse Mode Automatic Differentiation

## Topics discussed:
Reverse Mode Automatic Differentiation for Addition and Multiplication operations on Scalars.

## Example
Consider a simple example.
#### Forward Pass

$$a = 5$$

$$b = 3$$

$$c = a + b = 8$$

$$d = c * b = 24$$

#### Backward Pass

$$\frac{\partial d}{\partial a} = b = 3$$
$$\frac{\partial d}{\partial b} = a + 2b = 11$$
$$\frac{\partial d}{\partial c} = b = 3$$

### Automatic Differentiation
- Importing the library.

    ```julia
    using jac
    ```
- Initialising variables

    ```julia
    a = Tensor(5)
    b = Tensor(3)
    ```
- Forward Pass (setting up graph)

    ```julia
    c = a + b
    d = c * b
    ```
- Backward Pass (getting gradients)

    ```julia
    ∂d = jac.autograd(d)
    ```
    - To get $\frac{\partial d}{\partial a}$
        ```julia
        ∂d[a]
        ```
    - To get $\frac{\partial d}{\partial b}$
        ```julia
        ∂d[b]
        ```
    - To get $\frac{\partial d}{\partial c}$
        ```julia
        ∂d[c]
        ```
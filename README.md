# JAC

[![Documenter](https://github.com/tgautam03/jac/actions/workflows/Documenter.yml/badge.svg)](https://github.com/tgautam03/jac/actions/workflows/Documenter.yml)
[![Runtests](https://github.com/tgautam03/jac/actions/workflows/Runtests.yml/badge.svg)](https://github.com/tgautam03/jac/actions/workflows/Runtests.yml)

**JAC**obian is a lightweight Automatic Differentiation library written in Julia.

## Linear Regression using JAC

```julia
# Importing Packages Needed
using jac
using Plots
```

**Generating Synthetic Dataset**

```julia
# Generating Synthetic Data
x = range(-5, 5, 50); # X range
y = range(-5, 5, 50); # Y range

grid = Iterators.product(x, y); # Grid Operator
X = reshape(collect.(grid), 50*50); # Getting Grid points
X = reduce(hcat,X)' # Reshaping grid points to a Matrix


W = rand(2,1) # True Weights
b = rand(1,1) # True Bias

Y = X * W .+ b; # True Labels
```

**Moving data to `Tensor`s**

```julia
X = Tensor(Array(X)); # Converting Arrays to Tensors
Y = Tensor(Array(Y)); # Converting Arrays to Tensors
```

**Initialising Weights and Biases**

```julia
EPOCHS = 500;

W_init = Tensor(rand(2,1)*2) # Initial Weights
b_init = Tensor(zeros(1,1)) # Initial Bias
```

**Updating Parameters**

```julia
MSE = zeros(EPOCHS); # Array to store MSE at each epoch
for i = 1:EPOCHS # Iterations for Gradient Descent
    Y_pred = X*W_init + b_init; # Prediction
    
    mse = sum((Y_pred - Y)^2)/size(X)[1]; # Mean Squared Error
    println("Epoch $i; MSE: $(mse.val)")
    MSE[i] = mse.val

    # Updating Weights and Bias
    gradients = jac.grad(mse);

    W_init.val = W_init.val - 0.01*gradients[W_init];
    b_init.val = b_init.val - 0.01*gradients[b_init];
end
```

**MSE vs Epochs**

```julia
plot(MSE, xlabel="Epochs", ylabel="MSE", title="Training Curve", labels="mean squared error")
```

![mse](docs/src/img/MSE.png)
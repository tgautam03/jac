# Importing Packages Needed
using jac
using Plots

# Generating Synthetic Data
x = range(-5, 5, 50); # X range
y = range(-5, 5, 50); # Y range

grid = Iterators.product(x, y); # Grid Operator
X = reshape(collect.(grid), 50*50); # Getting Grid points
X = reduce(hcat,X)' # Reshaping grid points to a Matrix


W = rand(2,1) # True Weights
b = rand(1,1) # True Bias

Y = X * W .+ b; # True Labels


# Linear Single Layered Neural Network (Linear Regression)

X = Tensor(Array(X));
Y = Tensor(Array(Y));

EPOCHS = 200;

W_init = Tensor(rand(2,1)*5)
b_init = Tensor(zeros(1,1))

MSE = zeros(EPOCHS);
for i = 1:EPOCHS
    Y_pred = X*W_init + b_init;
    mse = sum((Y_pred - Y)^2)/size(X)[1];
    println("Epoch $i; MSE: $(mse.val)")
    MSE[i] = mse.val

    # Updating Weights and Bias
    gradients = jac.grad(mse);

    W_init.val = W_init.val - 0.01*gradients[W_init];
    b_init.val = b_init.val - 0.01*gradients[b_init];
end

plot(MSE, xlabel="Epochs", ylabel="MSE", title="Training Curve", labels="mean squared error")

W_Err = abs.(W_init.val - W)
b_Err = abs.(b_init.val - b)
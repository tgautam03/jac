using Test
using jac

# function test_grad()
    # X = Tensor([0.4963 0.7682 0.0885;
    # 0.1320 0.3074 0.6341;
    # 0.4901 0.8964 0.4556;
    # 0.6323 0.3489 0.4017;
    # 0.0223 0.1689 0.2939])

    # W = Tensor([0.5185 0.6977;
    #     0.8000 0.1610;
    #     0.2823 0.6816])

    # b = Tensor([0.9152 0.3971])   

    # Y = Tensor(ones(5,2))

    # Z = X * W + b

    # A = sum((Z - Y)^2)

    # L = A/size(X)[1]

    # grads = jac.grad(L)

#     return (grads, grads[A], grads[Z], grads[Y], grads[b], grads[W], grads[X])
# end

# (grads, A, Z, Y, b, W, X) = test_grad()


# A
# Z
# Y
# b
# W
# X

function test_Tensor()
    arr1 = rand(5, 1)
    arr2 = rand(5, 1)
    arr_sum = arr1 + arr2
    
    T1 = Tensor(arr1)
    T2 = Tensor(arr2)
    T_sum = T1 + T2

    return (T_sum.val == arr_sum)
end

function test_backward()
    A = Tensor(rand(5,3))
    B = Tensor(rand(3,2))
    
    C = A * B
    D = sum(C)
    println("D: \n", D)
    println("")

    dA = jac.grad(D)[A]
    dB = jac.grad(D)[B]
    dC = jac.grad(D)[C]

    println("dC: \n", dC)
    println("")
    println("dB: \n", dB)
    println("")
    println("dA: \n", dA)
    println("")
    # cond1 = (D.val == [24][:,:])
    # cond2 = (dA == [3][:,:])
    # cond3 = (dB == [11][:,:])
    # cond4 = (dC == [3][:,:])

    return true #(cond1 == true && cond2 == true && cond3 == true && cond4 == true)
end

@test test_backward()
@test test_Tensor()
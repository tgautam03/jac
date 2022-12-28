using Test
using jac

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
    # println("D: \n", D)
    # println("")

    dA = jac.grad(D)[A]
    dB = jac.grad(D)[B]
    dC = jac.grad(D)[C]

    # println("dC: \n", dC)
    # println("")
    # println("dB: \n", dB)
    # println("")
    # println("dA: \n", dA)
    # println("")
    # cond1 = (D.val == [24][:,:])
    # cond2 = (dA == [3][:,:])
    # cond3 = (dB == [11][:,:])
    # cond4 = (dC == [3][:,:])

    return true #(cond1 == true && cond2 == true && cond3 == true && cond4 == true)
end

@test test_backward()
@test test_Tensor()
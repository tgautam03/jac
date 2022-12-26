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
    A = Tensor([5][:,:])
    B = Tensor([3][:,:])
    
    C = A + B
    D = B * C

    dA = grad(D)[A]
    dB = grad(D)[B]
    dC = grad(D)[C]

    cond1 = (D.val == [24][:,:])
    cond2 = (dA == [3][:,:])
    cond3 = (dB == [11][:,:])
    cond4 = (dC == [3][:,:])

    return (cond1 == true && cond2 == true && cond3 == true && cond4 == true)
end

@test test_Tensor()
@test test_backward()

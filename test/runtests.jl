using Test
using jac

function test_Tensor()
    arr1 = rand(5, 1)
    arr2 = rand(5, 1)
    arr_sum = arr1 + arr2
    
    T1 = Tensor(arr1)
    T2 = Tensor(arr2)
    T_sum = T1 + T2

    cond = (T_sum == arr_sum)
end

@test test_Tensor()

using Test
using JAC

function test1()
    a = Tensor(5)
    b = Tensor(3)
    c = a + b
    d = b * c
    ∂d = JAC.autograd(d)

    cond1 = (c.val == 8)
    cond2 = (d.val == 24)
    cond3 = (∂d[a].val == 3)
    cond4 = (∂d[b].val == 11)
    cond5 = (∂d[c].val == 3)

    return (cond1 && cond2 && cond3 && cond4 && cond5)
end

@test test1()
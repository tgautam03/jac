using Test
using jac

function test1()
    # Primitive Operations
    A = rand()
    B = rand()
    C = A + B
    D = C * A

    # Variable Operations
    A_ = Variable(A)
    B_ = Variable(B)
    C_ = A_ + B_
    D_ = C_ * A_

    grads = autograd(D_)
    println(grads[A_].value)
    println(grads[B_].value)

    return (C_.value == C) && (D_.value == D)
end


@test test1()
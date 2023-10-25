using Test
using jac

function test1()
    # Primitive Operations
    A = rand()
    B = rand()
    C = A + B
    D = C * B

    # Variable Operations
    A_ = Variable(A)
    B_ = Variable(B)
    C_ = A_ + B_
    D_ = C_ * B_

    return (C_.value == C) && (D_.value == D)
end


@test test1()
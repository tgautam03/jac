mutable struct Variable
    value::Number       # Value of this Variable
    parents::Vector     # Who created this Variable
    chain_rules::Vector # Functions representing the chain rules

    # Constructor for creating Leaf Nodes
    function Variable(value::Number)
        new(value, [], [])
    end

    # Constructor for creating Intermediate Nodes
    function Variable(value::Number, parents::Vector{Variable}, chain_rules::Vector)
        new(value, parents, chain_rules)
    end
end
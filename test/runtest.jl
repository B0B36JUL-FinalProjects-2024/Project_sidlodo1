module Tests
using Test

include("../src/ClassifierModels.jl")
include("../src/Trees.jl")

@testset "Tests" begin
    include("test_utils.jl")
    include("test_logreg.jl")
    include("test_trees.jl")
end
end

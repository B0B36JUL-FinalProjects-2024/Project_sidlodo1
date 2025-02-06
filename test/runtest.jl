using Test

@testset "Tests" begin
    include("test_utils.jl")
    include("test_logreg.jl")
    include("test_trees.jl")
end

module test_Trees

using Test

include("../src/Trees.jl")
import .Trees

@testset "Trees Tests" begin

    X = [1 5;
        2 6;
        1 4;
        1 6;
        3 4;
        2 4]
    y = [0, 1, 0, 1, 0, 1]

    leaf1 = Trees.TreeNode(-1, 0.0, nothing, nothing, 0.0)
    leaf2 = Trees.TreeNode(-1, 0.0, nothing, nothing, 1.0)
    tree = Trees.TreeNode(1, 4.0, leaf1, leaf2, nothing)

    @testset "gini_impurity" begin
        impurity = Trees.gini_impurity(y)
        @test impurity == 0.5
    end

    @testset "weighted_gini" begin
        y_left = [0, 0, 1, 1]
        y_right = [0, 1]
        w_gini = Trees.weighted_gini(y_left, y_right)
        @test isapprox(w_gini, 0.5)
    end

    @testset "find_best_split" begin
        best_feature, best_threshold, best_gini, _, _ = Trees.find_best_split(X, y)
        @test best_feature == 2
        @test best_threshold == 5
        @test best_gini == 0.25
    end

    @testset "build_tree" begin
        tree_tmp = Trees.build_tree(X, y, :gradboost, 2, 2, 2)
        @test tree_tmp.feature_idx == 2
        @test tree_tmp.threshold == 5.0
    end

    @testset "Predict Tree" begin
        @test Trees.predict_tree(tree, [3.0, 2.0]) == 0.0
        @test Trees.predict_tree(tree, [5.0, 2.0]) == 1.0
    end

    @testset "get_value" begin
        y = [1, 1, 0, 0, 1]
        @test Trees.get_value(y, :randomforest) == 1
        @test Trees.get_value(y, :gradboost) == 0.6
    end
end

end

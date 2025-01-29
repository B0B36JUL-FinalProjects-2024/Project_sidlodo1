module Decision

using LinearAlgebra
using Statistics
using DecisionTree

include("Trees.jl")
import .Trees

mutable struct GradientBoostedTrees
    trees::Vector{DecisionTree.DecisionTreeRegressor}
    learning_rate::Float64
end

function build_tree(X::Matrix, residuals::Vector, max_depth::Int, min_samples_split::Int)
    model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split)
    model = DecisionTree.fit(model, X, residuals)
    fit!(model, X, residuals)
    return model
end

function train(X::Matrix, y::Vector; n_trees::Int=100, learning_rate::Float64=0.1, max_depth::Int=3, min_samples_split::Int=2)
    m, _ = size(X)
    trees = Vector{DecisionTree.DecisionTreeRegressor}()
    
    initial_prediction = mean(y)
    predictions = fill(initial_prediction, m)
    
    for _ in 1:n_trees
        residuals = y .- predictions
        tree = build_tree(X, residuals, max_depth, min_samples_split)
        push!(trees, tree)
        for i in 1:m
            predictions[i] += learning_rate * predict(tree, X[i, :])
        end
    end
    
    return GradientBoostedTrees(trees, learning_rate)
end

function predict(model::GradientBoostedTrees, X::Matrix; threshold::Float64=0.5)
    m, _ = size(X)
    predictions = fill(0.0, m)
    
    for tree in model.trees
        for i in 1:m
            predictions[i] += model.learning_rate * predict(tree, X[i, :])
        end
    end

    return predictions .>= threshold
end

end

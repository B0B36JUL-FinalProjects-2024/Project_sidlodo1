module GradBoost

export train!, predict

using LinearAlgebra
using Statistics

include("ClassifierModels.jl")
import .ClassifierModels

include("Trees.jl")
import .Trees

mutable struct GradBoostModel <: ClassifierModels.Model
    n_trees::Int
    max_depth::Int
    min_samples_split::Int
    max_features::Int
    trees::Vector{Trees.TreeNode}
    lr::Float64
    initial_prediction::Float64

    function GradBoostModel(;n_trees::Int=10, lr::Float64=0.1, max_depth::Int=5, min_samples_split::Int=2, max_features::Int=2, trees::Vector=Vector{Trees.TreeNode}(), initial_prediction::Float64=0.0)
        new(n_trees, max_depth, min_samples_split, max_features, trees, lr, initial_prediction)
    end
end

"""
Trains a gradient boosting trees model where n_trees are built sequentially on the errors of the previous tree.
"""
function train!(X::Matrix, y::Vector, model::GradBoostModel)
    m = size(X, 1)
    model.initial_prediction = mean(y)
    predictions = fill(model.initial_prediction, m)
    
    for _ in 1:model.n_trees
        errors = y .- predictions
        tree = Trees.build_tree(X, errors, Trees.GradBoostMethod(), model.max_depth, model.min_samples_split, model.max_features)
        push!(model.trees, tree)
        for i in 1:m
            predictions[i] += model.lr * Trees.predict_tree(tree, X[i, :])
        end
    end
end

"""
Predicts the class of the input data using the gradient boosting trees model where the sum of predictions from all trees is taken.
"""
function predict(model::GradBoostModel, X::Matrix; threshold::Float64=0.5)
    m = size(X, 1)
    predictions = fill(model.initial_prediction, m)
    
    for tree in model.trees
        for i in 1:m
            predictions[i] += model.lr * Trees.predict_tree(tree, X[i, :])
        end
    end

    return predictions .>= threshold
end

end

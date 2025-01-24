module GradBoost

using LinearAlgebra
using Statistics

include("Trees.jl")
import .Trees

mutable struct GradientBoostedTrees
    trees::Vector{Trees.TreeNode}
    learning_rate::Float64
end

function build_tree(X::Matrix, residuals::Vector, max_depth::Int, min_samples_split::Int)
    if max_depth == 0 || length(residuals) < min_samples_split
        return Trees.TreeNode(-1, 0.0, nothing, nothing, mean(residuals))  # Leaf node with mean residual
    end

    # Find the best split using Trees.find_best_split
    best_split, left_mask, right_mask = Trees.find_best_split(X, residuals)
    feature_index, threshold = best_split

    if feature_index == -1
        return Trees.TreeNode(-1, 0.0, nothing, nothing, mean(residuals))  # No valid split, create a leaf
    end

    # Recursively build left and right subtrees
    left_tree = build_tree(X[left_mask, :], residuals[left_mask], max_depth - 1, min_samples_split)
    right_tree = build_tree(X[right_mask, :], residuals[right_mask], max_depth - 1, min_samples_split)

    return Trees.TreeNode(feature_index, threshold, left_tree, right_tree, nothing)
end

# Train Gradient Boosted Trees
function train(X::Matrix, y::Vector; n_trees::Int=100, learning_rate::Float64=0.1, max_depth::Int=3, min_samples_split::Int=2)
    m, _ = size(X)
    trees = Vector{Trees.TreeNode}()
    
    initial_prediction = mean(y)
    predictions = fill(initial_prediction, m)
    
    for _ in 1:n_trees
        residuals = y .- predictions
        tree = build_tree(X, residuals, max_depth, min_samples_split)
        push!(trees, tree)
        for i in 1:m
            predictions[i] += learning_rate * Trees.predict_tree(tree, X[i, :])
        end
    end
    
    return GradientBoostedTrees(trees, learning_rate)
end

function predict(model::GradientBoostedTrees, X::Matrix; threshold::Float64=0.5)
    m, _ = size(X)
    predictions = fill(0.0, m)
    
    for tree in model.trees
        for i in 1:m
            predictions[i] += model.learning_rate * Trees.predict_tree(tree, X[i, :])
        end
    end

    return predictions .>= threshold
end

end

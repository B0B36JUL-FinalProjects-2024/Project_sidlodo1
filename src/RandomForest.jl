module RandomForest

using Random
using DataFrames
using StatsBase

include("Trees.jl")
import .Trees

function build_tree(X::Matrix, y::Vector, max_depth::Int, min_samples_split::Int, k::Int)
    if max_depth == 0 || size(X, 1) < min_samples_split
        return Trees.TreeNode(-1, 0.0, nothing, nothing, majority_class(y))
    end

    n_features = size(X, 2)
    selected_features = rand(1:n_features, k)

    best_split, left_mask, right_mask = find_best_split(X[:, selected_features], y)
    feature_index, threshold = best_split

    if feature_index == -1
        return Trees.TreeNode(-1, 0.0, nothing, nothing, majority_class(y))
    end

    left_tree = build_tree(X[left_mask, :], y[left_mask], max_depth - 1, min_samples_split, k)
    right_tree = build_tree(X[right_mask, :], y[right_mask], max_depth - 1, min_samples_split, k)

    return Trees.TreeNode(feature_index, threshold, left_tree, right_tree, nothing)
    # return Trees.TreeNode(selected_features[feature_index], threshold, left_tree, right_tree, nothing)
end

function train(X::Matrix, y::Vector; n_trees::Int=5, max_depth::Int=3, min_samples_split::Int=2, k::Int=2)
    trees = Vector{Trees.TreeNode}()
    
    for _ in 1:n_trees
        n_samples = size(X, 1)
        sampled_indices = rand(1:n_samples, n_samples)
        X_sample = X[sampled_indices, :]
        y_sample = y[sampled_indices]
        
        tree = Trees.build_tree(X_sample, y_sample, max_depth, min_samples_split, k)
        push!(trees, tree)
    end
    
    return trees
end

function predict(forest::Vector{Trees.TreeNode}, X::Union{Matrix, Vector})
    if isa(X, Vector)
        X = reshape(X, 1, :)
    end
    
    predictions = []
    
    for i in 1:size(X, 1)
        votes = [Trees.predict_tree(tree, X[i, :]) for tree in forest]
        push!(predictions, mode(votes))
    end
    
    return predictions
end

end

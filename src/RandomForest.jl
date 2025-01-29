module RandomForest

using Random
using DataFrames
using StatsBase

include("Trees.jl")
import .Trees

"""
Trains a random forest model where n_trees are built independently on a random subset of the data and features.
"""
function train(X::Matrix, y::Vector; n_trees::Int=5, max_depth::Int=3, min_samples_split::Int=2, max_features::Int=2)
    trees = Vector{Trees.TreeNode}()
    
    for _ in 1:n_trees
        n_samples = size(X, 1)
        sampled_indices = rand(1:n_samples, n_samples)
        X_sample = X[sampled_indices, :]
        y_sample = y[sampled_indices]
        
        tree = Trees.build_tree(X_sample, y_sample, :randomforest ,max_depth, min_samples_split, max_features)
        push!(trees, tree)
    end
    
    return trees
end

"""
Predicts the class of the input data using the random forest model where majority of votes from all trees is taken.
"""
function predict(forest::Vector{Trees.TreeNode}, X::Union{Matrix, Vector})
    if isa(X, Vector)
        X = reshape(X, 1, :)
    end
    
    predictions = []
    
    for i in axes(X, 1)
        votes = [Trees.predict_tree(tree, X[i, :]) for tree in forest]
        push!(predictions, mode(votes))
    end
    
    return predictions
end

end

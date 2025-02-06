module RandomForest

export train!, predict

using Random
using DataFrames
using StatsBase

include("ClassifierModels.jl")
import .ClassifierModels

include("Trees.jl")
import .Trees

mutable struct RandomForestModel <: ClassifierModels.Model
    n_trees::Int
    max_depth::Int
    min_samples_split::Int
    max_features::Int
    trees::Vector{Trees.TreeNode}

    function RandomForestModel(;n_trees::Int=10, max_depth::Int=5, min_samples_split::Int=2, max_features::Int=2, trees::Vector=Vector{Trees.TreeNode}())
        new(n_trees, max_depth, min_samples_split, max_features, trees)
    end
end

"""
Trains a random forest model where n_trees are built independently on a random subset of the data and features.
"""
function train!(X::Matrix, y::Vector, model::RandomForestModel)
    
    for _ in 1:model.n_trees
        n_samples = size(X, 1)
        sampled_indices = rand(1:n_samples, n_samples)
        X_sample = X[sampled_indices, :]
        y_sample = y[sampled_indices]
        
        tree = Trees.build_tree(X_sample, y_sample, Trees.RandomForestMethod(), model.max_depth, model.min_samples_split, model.max_features)
        push!(model.trees, tree)
    end
    
end

"""
Predicts the class of the input data using the random forest model where majority of votes from all trees is taken.
"""
function predict(model::RandomForestModel, X::Matrix)
    predictions = []
    
    for i in axes(X, 1)
        votes = [Trees.predict_tree(tree, X[i, :]) for tree in model.trees]
        push!(predictions, mode(votes))
    end
    
    return predictions
end

end

module RandomForest

export train!, predict

using Random
using DataFrames
using StatsBase

using ..ClassifierModels
using ..Trees

"""
Trains a random forest model where n_trees are built independently on a random subset of the data and features.
"""
function train!(X::Matrix, y::Vector, model::RandomForestModel)
    
    for _ in 1:model.n_trees
        n_samples = size(X, 1)
        sampled_indices = rand(1:n_samples, n_samples)
        X_sample = X[sampled_indices, :]
        y_sample = y[sampled_indices]
        
        tree = Trees.build_tree(X_sample, y_sample, RandomForestMethod(), model.max_depth, model.min_samples_split, model.max_features)
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

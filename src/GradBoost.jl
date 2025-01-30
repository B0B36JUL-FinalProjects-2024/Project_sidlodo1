module GradBoost

using LinearAlgebra
using Statistics

include("Trees.jl")
import .Trees

mutable struct GradientBoostedTrees
    trees::Vector{Trees.TreeNode}
    learning_rate::Float64
    initial_prediction::Float64
    n_trees::Int = 100, learning_rate::Float64 = 0.1, max_depth::Int = 3, min_samples_split::Int = 2, max_features::Int = 2

end

model = GradientBoostedTrees(n_trees::Int = 100, learning_rate::Float64 = 0.1, max_depth::Int = 3, min_samples_split::Int = 2, max_features::Int = 2)
train(model, X, y)
predict(model, X, y)

abstract type EasyClassifier end

model = GradientBoostedTrees()
function do_classification_with_report(model::Classifier, X, y)
    params = train(model, X, y)
    train!(model, X, y)

    y_pred = predict(model, X, y)

    acc = accuracy(y, y_pred)

    print(
        """
        Report:

        - model = $(model.name)
        - accuracy = 
        """
    )
    return y_pred
end



"""
Trains a gradient boosting trees model where n_trees are built sequentially on the errors of the previous tree.
"""
function train(X::Matrix, y::Vector, model::GradientBoostedTrees)
    m, _ = size(X)
    trees = Vector{Trees.TreeNode}()

    initial_prediction = mean(y)
    # initial_prediction = median(y)
    predictions = fill(initial_prediction, m)

    for _ in 1:n_trees
        errors = y .- predictions
        tree = Trees.build_tree(X, errors, :gradboost, max_depth, min_samples_split, max_features)
        push!(trees, tree)
        for i in 1:m
            predictions[i] += learning_rate * Trees.predict_tree(tree, X[i, :])
        end
    end

    return GradientBoostedTrees(trees, learning_rate, initial_prediction)
end

"""
Predicts the class of the input data using the gradient boosting trees model where the sum of predictions from all trees is taken.
"""
function predict(model::GradientBoostedTrees, X::Matrix; threshold::Float64=0.5)
    m, _ = size(X)
    predictions = fill(model.initial_prediction, m)

    for tree in model.trees
        for i in 1:m
            predictions[i] += model.learning_rate * Trees.predict_tree(tree, X[i, :])
        end
    end

    return predictions .>= threshold
end

end


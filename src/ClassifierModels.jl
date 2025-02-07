module ClassifierModels

export Classifier, LogRegOptimizationMethod, GradientDescentMethod, NewtonMethod, LogRegModel, EnsembleMethod, RandomForestMethod, GradBoostMethod, TreeNode, RandomForestModel, GradBoostModel, DecisionTree

abstract type Classifier end

struct LogRegModel <: Classifier
    name::String
    n_iters::Int
    function LogRegModel(;name="Logistic Regression",n_iters=20)
        new(name,n_iters)
    end
end

abstract type LogRegOptimizationMethod end

struct GradientDescentMethod <: LogRegOptimizationMethod
    name::String
    lr::Float64
    function GradientDescentMethod(;lr=0.01)
        new("Gradient Descent", lr)
    end
end

struct NewtonMethod <: LogRegOptimizationMethod
    name::String
    function NewtonMethod()
        new("Newton")
    end
end

abstract type DecisionTree <:Classifier end

mutable struct TreeNode
    feature_idx::Int
    threshold::Float64
    left::Union{TreeNode, Nothing}
    right::Union{TreeNode, Nothing}
    value::Union{Float64, Nothing}
end

abstract type EnsembleMethod end

struct RandomForestMethod <: EnsembleMethod end
struct GradBoostMethod <: EnsembleMethod end

mutable struct RandomForestModel <: DecisionTree
    name::String
    n_trees::Int
    max_depth::Int
    min_samples_split::Int
    max_features::Int
    trees::Vector{TreeNode}

    function RandomForestModel(;n_trees=10, max_depth=5, min_samples_split=2, max_features=2, trees=Vector{TreeNode}())
        new("Random Forest", n_trees, max_depth, min_samples_split, max_features, trees)
    end
end

mutable struct GradBoostModel <: DecisionTree
    name::String
    n_trees::Int
    max_depth::Int
    min_samples_split::Int
    max_features::Int
    trees::Vector{TreeNode}
    lr::Float64
    initial_prediction::Float64

    function GradBoostModel(;n_trees=10, lr=0.1, max_depth=5, min_samples_split=2, max_features=2, trees=Vector{TreeNode}(), initial_prediction::Float64=0.0)
        new("Gradient Boosting Trees", n_trees, max_depth, min_samples_split, max_features, trees, lr, initial_prediction)
    end
end

end

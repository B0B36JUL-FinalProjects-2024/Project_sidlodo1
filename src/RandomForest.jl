module RandomForest

using Random
using DataFrames
using StatsBase

mutable struct Tree
    feature_index::Int
    threshold::Float64
    left::Union{Tree, Nothing}
    right::Union{Tree, Nothing}
    value::Union{Int, Nothing}
end

function gini_impurity(y::Vector)
    classes = unique(y)
    impurity = 1.0
    for cls in classes
        p_cls = sum(y .== cls) / length(y)
        impurity -= p_cls^2
    end
    return impurity
end

function find_best_split(X::Matrix, y::Vector)
    n, m = size(X)
    best_gini = Inf
    best_split = (-1, 0.0)
    best_left_indices = falses(n)
    best_right_indices = falses(n)
    
    for j in 1:m
        feature_values = X[:, j]
        possible_splits = unique(feature_values)
        
        for split_value in possible_splits
            left_mask = feature_values .<= split_value
            right_mask = .!left_mask
            
            left_y = y[left_mask]
            right_y = y[right_mask]
            
            if length(left_y) > 0 && length(right_y) > 0
                gini_left = gini_impurity(left_y)
                gini_right = gini_impurity(right_y)
                weighted_gini = (length(left_y) / n) * gini_left + (length(right_y) / n) * gini_right
                
                if weighted_gini < best_gini
                    best_gini = weighted_gini
                    best_split = (j, split_value)
                    best_left_indices = left_mask
                    best_right_indices = right_mask
                end
            end
        end
    end

    return best_split, best_left_indices, best_right_indices
end

function build_tree(X::Matrix, y::Vector, max_depth::Int, min_samples_split::Int, k::Int)
    # Base case: if the depth is maxed out or there are not enough samples to split
    if max_depth == 0 || size(X, 1) < min_samples_split
        return Tree(-1, 0.0, nothing, nothing, majority_class(y))
    end

    n_features = size(X, 2)
    selected_features = rand(1:n_features, k)  # Randomly select k features

    # Find the best split
    best_split, left_mask, right_mask = find_best_split(X[:, selected_features], y)
    feature_index, threshold = best_split

    # If no valid split is found, return a leaf node
    if feature_index == -1
        return Tree(-1, 0.0, nothing, nothing, majority_class(y))
    end

    # Build the left and right subtrees recursively
    left_tree = build_tree(X[left_mask, :], y[left_mask], max_depth - 1, min_samples_split, k)
    right_tree = build_tree(X[right_mask, :], y[right_mask], max_depth - 1, min_samples_split, k)
    
    # Create a tree node and return it
    return Tree(feature_index, threshold, left_tree, right_tree, nothing)
end

function majority_class(y::Vector)
    return mode(y)
end

function train(X::Matrix, y::Vector; n_trees::Int=5, max_depth::Int=3, min_samples_split::Int=2, k::Int=2)
    trees = Vector{Tree}()
    
    for _ in 1:n_trees
        n_samples = size(X, 1)
        sampled_indices = rand(1:n_samples, n_samples)
        X_sample = X[sampled_indices, :]
        y_sample = y[sampled_indices]
        
        tree = build_tree(X_sample, y_sample, max_depth, min_samples_split, k)
        push!(trees, tree)
    end
    
    return trees
end

function predict_tree(tree::Tree, x::Vector)
    if tree.left === nothing && tree.right === nothing
        return tree.value
    end
    
    if x[tree.feature_index] <= tree.threshold
        return predict_tree(tree.left, x)
    else
        return predict_tree(tree.right, x)
    end
end

function predict(forest::Vector{Tree}, X::Union{Matrix, Vector})
    if isa(X, Vector)
        X = reshape(X, 1, :)
    end
    
    predictions = []
    
    for i in 1:size(X, 1)
        votes = [predict_tree(tree, X[i, :]) for tree in forest]
        push!(predictions, mode(votes))
    end
    
    return predictions
end

end

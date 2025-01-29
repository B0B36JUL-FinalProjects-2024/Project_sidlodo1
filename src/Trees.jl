module Trees

using Statistics
using StatsBase

mutable struct TreeNode
    feature_idx::Int
    threshold::Float64
    left::Union{TreeNode, Nothing}
    right::Union{TreeNode, Nothing}
    value::Union{Float64, Nothing}
end

"""
Calculates the Gini impurity for a set of labels.
"""
function gini_impurity(y::Vector)
    classes = unique(y)
    impurity = 1.0
    for cls in classes
        p_cls = sum(y .== cls) / length(y)
        impurity -= p_cls^2
    end
    return impurity
end

"""
Calculates the weighted Gini impurity for a split.
"""
function weighted_gini(left_y::Vector, right_y::Vector)
    n = length(left_y) + length(right_y)
    w_left = length(left_y) / n
    w_right = length(right_y) / n
    gini_left = gini_impurity(left_y)
    gini_right = gini_impurity(right_y)
    return w_left * gini_left + w_right * gini_right
end

"""
Find best features to split the data to minimize the Gini impurity.
"""
function find_best_split(X::Matrix, y::Vector)
    n, m = size(X)
    best_gini = Inf
    best_feature = -1
    best_threshold = 0.0
    best_left_idx = falses(n)
    best_right_idx = falses(n)
    
    for j in 1:m
        feature_values = X[:, j]
        possible_splits = unique(feature_values)
        
        for split_value in possible_splits
            left_feat = feature_values .<= split_value
            right_feat = .!left_feat
            
            left_y = y[left_feat]
            right_y = y[right_feat]
            
            if length(left_y) > 0 && length(right_y) > 0
                w_gini = weighted_gini(left_y, right_y)
                if w_gini < best_gini
                    best_gini = w_gini
                    best_feature = j
                    best_threshold = split_value
                    best_left_idx = left_feat
                    best_right_idx = right_feat
                end
            end
        end
    end
    return best_feature, best_threshold, best_gini, best_left_idx, best_right_idx
end

"""
Returns the value to predict for the leaf node.
"""
function get_value(y::Vector, method::Symbol)
    if method == :gradboost
        return mean(y)
    elseif method == :randomforest
        return mode(y)
    else
        throw(ArgumentError("Invalid method"))
    end
end

"""
Predicts the class of the input data using the tree model.
"""
function predict_tree(tree::Trees.TreeNode, x::Vector)
    if tree.left === nothing && tree.right === nothing
        return tree.value
    end
    
    if x[tree.feature_idx] <= tree.threshold
        return predict_tree(tree.left, x)
    else
        return predict_tree(tree.right, x)
    end
end

"""
Builds a decision tree using the CART algorithm.
"""
function build_tree(X::Matrix, y::Vector, method::Symbol, max_depth::Int, min_samples_split::Int, max_features::Int)
    if max_depth == 0 || length(y) < min_samples_split
        return TreeNode(-1, 0.0, nothing, nothing, get_value(y, method))
    end

    n_features = size(X, 2)
    selected_features = rand(1:n_features, max_features)

    feature_idx, threshold, _, left_feat, right_feat = find_best_split(X[:, selected_features], y)
    if feature_idx == -1
        return TreeNode(-1, 0.0, nothing, nothing, get_value(y, method))
    end

    feature_idx = selected_features[feature_idx]

    left_tree = build_tree(X[left_feat, :], y[left_feat], method, max_depth - 1, min_samples_split, max_features)
    right_tree = build_tree(X[right_feat, :], y[right_feat], method, max_depth - 1, min_samples_split, max_features)

    return TreeNode(feature_idx, threshold, left_tree, right_tree, nothing)
end

end

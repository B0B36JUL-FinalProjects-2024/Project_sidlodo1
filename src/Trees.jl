module Trees

using Statistics

mutable struct TreeNode
    feature_index::Int
    threshold::Float64
    left::Union{TreeNode, Nothing}
    right::Union{TreeNode, Nothing}
    value::Union{Float64, Nothing}
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

function majority_class(y::Vector)
    return mode(y)
end

function predict_tree(tree::Trees.TreeNode, x::Vector)
    if tree.left === nothing && tree.right === nothing
        return tree.value
    end
    
    if x[tree.feature_index] <= tree.threshold
        return predict_tree(tree.left, x)
    else
        return predict_tree(tree.right, x)
    end
end

end

module LogReg

using LinearAlgebra
using Statistics

"""
Applies the sigmoid function to a scalar or vector.
"""
function sigmoid(x)
    return 1.0 / (1.0 + exp(-x))
end

"""
Normalizes a matrix.
"""
function normalize(X::Matrix)
    means = mean(X, dims=1)
    stds = std(X, dims=1)
    stds[stds .== 0] .= 1
    return (X .- means) ./ stds
end

# function compute_loss(labels::Vector, predictions::Vector)
#     return -mean(labels .* log.(predictions) .+ (1 .- labels) .* log.(1 .- predictions))
# end

"""
Performs one step of gradient descent with regularization and learning rate decay.
"""
function grad_descent(X::Matrix, y::Vector, w::Vector, b::Float64, n::Int, lr::Float64)
    h = sigmoid.(X * w .+ b)
    dz = h - y
    dw = (1/n) * X' * dz
    db = (1/n) * sum(dz)
    w -= lr * dw
    b -= lr * db
    return w, b
end

"""
Performs one step of Newton's method for optimizing weights.
"""
function newton(X::Matrix, y::Vector, X_mult::Vector, w::Vector)
    h = sigmoid.(X * w)
    grad = X' * (h .- y) / size(X,1)
    hess = h .* (1 .- h) .* X_mult |> mean
    w -= hess \ grad
    return w
end

"""
Trains a logistic regression model using gradient descent or Newton's method.
"""
function train(X::Matrix, y::Vector; lr::Float64=0.01, n_iters::Int=100, method::Symbol=:grad_descent)
    n, m = size(X)
    X = normalize(X)
    w = zeros(m)
    b = 0.0

    if method == :grad_descent
        for _ in 1:n_iters
            w, b = grad_descent(X, y, w, b, n, lr)
        end
        return w, b
    else
        X_mult = [row*row' for row in eachrow(X)]
        for _ in 1:n_iters
            w = newton(X, y, X_mult, w)
        end
        return w
    end
end

function predict(w::Vector, b::Float64, X::Matrix)
    X = normalize(X)
    probabilities = sigmoid.(X * w .+ b)
    return probabilities .>= 0.5
end

function predict(w::Vector, X::Matrix)
    X = normalize(X)
    probabilities = sigmoid.(X * w)
    return probabilities .>= 0.5
end

end

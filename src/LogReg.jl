module LogReg

export train, predict

using LinearAlgebra
using Statistics

using ..ClassifierModels

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

"""
Performs one step of Gradient Descent method with learning rate.
"""
function update!(X::Matrix, y::Vector, w::Vector, b::Float64, method::GradientDescentMethod)
    n = size(X, 1)
    h = sigmoid.(X * w .+ b)
    dz = h - y
    dw = (1/n) * X' * dz
    db = (1/n) * sum(dz)
    w -= method.lr * dw
    b -= method.lr * db
    return w, b
end

"""
Performs one step of Newton method.
"""
function update!(X::Matrix, y::Vector, w::Vector, method::NewtonMethod)
    X_mult = [row*row' for row in eachrow(X)]
    h = sigmoid.(X * w)
    grad = X' * (h .- y) / size(X,1)
    hess = h .* (1 .- h) .* X_mult |> mean
    w -= hess \ grad
    return w
end

"""
Trains a logistic regression model using Gradient Descent method.
"""
function train(X::Matrix, y::Vector, model::LogRegModel, method::GradientDescentMethod)
    X = normalize(X)
    w = zeros(size(X, 2))
    b = 0.0

    for _ in 1:model.n_iters
        w, b = update!(X, y, w, b, method)
    end

    return w, b
end

"""
Trains a logistic regression model using Newton method.
"""
function train(X::Matrix, y::Vector, model::LogRegModel, method::NewtonMethod)
    X = normalize(X)
    w = zeros(size(X, 2))

    for _ in 1:model.n_iters
        w = update!(X, y, w, method)
    end

    return w
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

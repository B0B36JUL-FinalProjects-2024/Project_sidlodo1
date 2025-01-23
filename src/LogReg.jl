module LogReg

using LinearAlgebra
using Statistics

function sigmoid(x)
    return 1.0 / (1.0 + exp(-x))
end

function normalize(X::Matrix)
    return (X .- mean(X, dims=1)) ./ std(X, dims=1)
end

function train(X::Matrix, y::Vector; lr::Float64=0.01, n_iters::Int=1000, λ::Float64=0.01, decay_rate::Float64=0.001)
    m, n = size(X)
    X = normalize(X)
    X = hcat(ones(m), X)
    θ = zeros(n + 1)

    for i in 1:n_iters
        z = X * θ
        h = sigmoid.(z)
        gradient = (1/m) * X' * (h - y) + (λ/m) * vcat(0, θ[2:end])
        θ -= lr * gradient
        lr = lr / (1 + decay_rate * i)

    end

    return θ
end

function predict(θ::Vector, X::Matrix; threshold::Float64=0.5)
    X = normalize(X)
    X = hcat(ones(size(X, 1)), X)
    probabilities = sigmoid.(X * θ)
    return probabilities .>= threshold
end

end

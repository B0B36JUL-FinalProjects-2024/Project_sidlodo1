module SurvivalPrediction

using CSV
using DataFrames
using Random
using Statistics
using Plots

include("../src/Utils.jl")
import .Utils

# import models:
include("RandomForest.jl")
include("LogReg.jl")
include("GradBoost.jl")
import .RandomForest
import .LogReg
import .GradBoost

function run_logreg(X_trn::Matrix, y_trn::Vector, tst_data::Matrix; lr::Float64=0.01, n_iters::Int=100, method::Symbol=:grad_descent)
    if method == :newton
        w = LogReg.train(X_trn, y_trn; lr=lr, n_iters=n_iters, method=:newton)
        predictions = LogReg.predict(w, tst_data)
    elseif method == :grad_descent
        w, b = LogReg.train(X_trn, y_trn; lr=lr, n_iters=n_iters, method=:grad_descent)
        predictions = LogReg.predict(w, b, tst_data)
    else
        throw(ArgumentError("Invalid method"))
    end
    return predictions
end

function run_randforest(X_trn::Matrix, y_trn::Vector, tst_data::Matrix; n_trees::Int=5, max_depth::Int=3, min_samples_split::Int=2, max_features::Int=2)
    model = RandomForest.train(X_trn, y_trn; n_trees=n_trees, max_depth=max_depth, min_samples_split=min_samples_split, max_features=max_features)
    predictions = RandomForest.predict(model, tst_data)
    return predictions
end

function run_gradboost(X_trn::Matrix, y_trn::Vector, tst_data::Matrix; n_trees::Int=100, learning_rate::Float64=0.1, max_depth::Int=3, min_samples_split::Int=2, max_features::Int=2)
    model = GradBoost.train(X_trn, y_trn; n_trees=n_trees, learning_rate=learning_rate, max_depth=max_depth, min_samples_split=min_samples_split, max_features=max_features)
    predictions = GradBoost.predict(model, tst_data)
    return predictions
end

"""
Plots the accuracy of a logistic regression model over a range of iterations with different learning rates.
"""
function plot_logreg_acc(X_trn, y_trn, X_tst, y_tst, param_set)
    plot()
    for params in param_set
        accuracies = Float64[]
        for iter in 1:params[:n_iters]
            w = LogReg.train(X_trn, y_trn; lr=params[:lr], n_iters=iter, method=:newton)
            predictions = LogReg.predict(w, X_tst)
            accuracy = Utils.classify_predictions(predictions, y_tst)
            push!(accuracies, accuracy)
        end
        plot!(1:params[:n_iters], accuracies, label="Newton: lr=$(params[:lr]), n_iters=$(params[:n_iters])")
        
        accuracies = Float64[]
        for iter in 1:params[:n_iters]
            w, b = LogReg.train(X_trn, y_trn; lr=params[:lr], n_iters=iter)
            predictions = LogReg.predict(w, b, X_tst)
            accuracy = Utils.classify_predictions(predictions, y_tst)
            push!(accuracies, accuracy)
        end
        plot!(1:params[:n_iters], accuracies, label="Grad descent: lr=$(params[:lr]), n_iters=$(params[:n_iters])")
    end
    xlabel!("Iterations")
    ylabel!("Accuracy")
    title!("Logistic Regression accuracy")
end

function plot_trees(X_trn, y_trn, X_tst, y_tst, param_set)
    plot()
    for params in param_set
        accuracies = Float64[]
        for iter in 1:params[:n_trees]
            model = RandomForest.train(X_trn, y_trn; n_trees=iter, max_depth=params[:depth], max_features=params[:features])
            predictions = RandomForest.predict(model, X_tst)
            accuracy = Utils.classify_predictions(predictions, y_tst)
            push!(accuracies, accuracy)
        end
        plot!(1:params[:n_trees], accuracies, label="Random forest: depth=$(params[:depth]), features=$(params[:features])")
        
        accuracies = Float64[]
        for iter in 1:params[:n_trees]
            model = GradBoost.train(X_trn, y_trn; n_trees=iter, max_depth=params[:depth], max_features=params[:features])
            predictions = GradBoost.predict(model, X_tst)
            accuracy = Utils.classify_predictions(predictions, y_tst)
            push!(accuracies, accuracy)
        end
        plot!(1:params[:n_trees], accuracies, label="Grad boosting: depth=$(params[:depth]), features=$(params[:features])")
    end
    xlabel!("Iterations")
    ylabel!("Accuracy")
    title!("Decision trees accuracy")
end

end

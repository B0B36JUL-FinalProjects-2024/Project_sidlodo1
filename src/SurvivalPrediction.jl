module SurvivalPrediction

using CSV
using DataFrames
using Random
using Statistics
using Plots

include("../src/Utils.jl")
import .Utils

# import models:
include("ClassifierModels.jl")
import .ClassifierModels
const CM = ClassifierModels
include("LogReg.jl")
import .LogReg
const LR = LogReg
include("RandomForest.jl")
import .RandomForest
const RF = RandomForest
include("GradBoost.jl")
import .GradBoost
const GB = GradBoost


function get_prediction(model::LR.LogRegModel, method::LR.GradientDescentMethod, X_trn::Matrix, y_trn::Vector, X_tst::Matrix)
    w, b = LR.train(X_trn, y_trn, model, method)
    predictions = LR.predict(w, b, X_tst)
    return predictions
end

function get_prediction(model::LR.LogRegModel, method::LR.NewtonMethod, X_trn::Matrix, y_trn::Vector, X_tst::Matrix)
    w = LR.train(X_trn, y_trn, model, method)
    predictions = LR.predict(w, b, X_tst)
    return predictions
end

function get_prediction(model::RF.RandomForestModel, X_trn::Matrix, y_trn::Vector, X_tst::Matrix)
    RF.train!(X_trn, y_trn, model)
    predictions = RF.predict(model, X_tst)
    return predictions
end

function get_prediction(model::GB.GradBoostModel, X_trn::Matrix, y_trn::Vector, X_tst::Matrix)
    GB.train!(X_trn, y_trn, model)
    predictions = GB.predict(model, X_tst)
    return predictions
end

"""
Plots the accuracy of a logistic regression model over a range of iterations with different learning rates.
"""
function plot_logreg_acc(X_trn, y_trn, X_tst, y_tst, param_set)
    plot()
    for params in param_set
        accuracies_newton = Float64[]
        accuracies_grad = Float64[]
        
        for iter in 1:params[:n_iters]
            model = LR.LogRegModel(n_iters=iter)
            # Newton method
            method_newton = LR.NewtonMethod()
            predictions_newton = get_prediction(model, method_newton, X_trn, y_trn, X_tst)
            accuracy_newton = Utils.classify_predictions(predictions_newton, y_tst)
            push!(accuracies_newton, accuracy_newton)
            
            # Gradient descent
            method_grad = LR.GradientDescentMethod(params[:lr])
            predictions_grad = get_prediction(model, method_grad, X_trn, y_trn, X_tst)
            accuracy_grad = Utils.classify_predictions(predictions_grad, y_tst)
            push!(accuracies_grad, accuracy_grad)
        end
        
        plot!(1:params[:n_iters], accuracies_newton, label="Newton: lr=$(params[:lr]), n_iters=$(params[:n_iters])")
        plot!(1:params[:n_iters], accuracies_grad, label="Grad descent: lr=$(params[:lr]), n_iters=$(params[:n_iters])")
    end
    xlabel!("Iterations")
    ylabel!("Accuracy")
    title!("Logistic Regression accuracy")
end

function plot_trees(X_trn, y_trn, X_tst, y_tst, param_set)
    plot()
    for params in param_set
        accuracies_rf = Float64[]
        accuracies_gb = Float64[]
        
        for iter in 1:params[:n_trees]
            # Random Forest
            predictions_rf = run_randforest(X_trn, y_trn, X_tst; n_trees=iter, max_depth=params[:depth], max_features=params[:features])
            accuracy_rf = Utils.classify_predictions(predictions_rf, y_tst)
            push!(accuracies_rf, accuracy_rf)
            
            # Gradient Boosting
            predictions_gb = run_gradboost(X_trn, y_trn, X_tst; n_trees=iter, max_depth=params[:depth], max_features=params[:features], lr=params[:lr])
            accuracy_gb = Utils.classify_predictions(predictions_gb, y_tst)
            push!(accuracies_gb, accuracy_gb)
        end
        
        plot!(1:params[:n_trees], accuracies_rf, label="Random forest: depth=$(params[:depth]), features=$(params[:features])")
        plot!(1:params[:n_trees], accuracies_gb, label="Grad boosting: depth=$(params[:depth]), features=$(params[:features])")
    end
    xlabel!("Iterations")
    ylabel!("Accuracy")
    title!("Decision trees accuracy")
end

end

module SurvivalPrediction

using CSV
using DataFrames
using Random
using Statistics

# import models:
include("RandomForest.jl")
include("LogReg.jl")
include("GradBoost.jl")
using .RandomForest
using .LogReg
using .GradBoost

function process_data(trn_data_path::String)
    data = CSV.File(trn_data_path) |> DataFrame

    data.Sex .= (data.Sex .== "male") .+ 0  # Example: 0 for male, 1 for female

    return data
end

function run_randforest(trn_data::DataFrame, tst_data::DataFrame)
    # train model and return trained model
    trn_model = RandomForest.train(trn_data)
    # predict on test data
    predictions = RandomForest.predict(trn_model, tst_data)
    return predictions
end

# Placeholder function to run Logistic Regression model (replace with actual code)
function run_logreg(trn_data::DataFrame, tst_data::DataFrame)
    # train model and return trained model
    trn_model = LogReg.train(trn_data)
    # predict on test data
    predictions = LogReg.predict(trn_model, tst_data)
    return predictions
end

# Placeholder function to run Gradient Boosting model (replace with actual code)
function run_gradboost(trn_data::DataFrame, tst_data::DataFrame)
    # train model and return trained model
    trn_model = GradBoost.train(trn_data)
    # predict on test data
    predictions = GradBoost.predict(trn_model, tst_data)
    return predictions
end


function run_models(file_path::String)
    trn_data = process_trn_data(file_path)
    tst_data = process_tst_data(file_path)

    rf_predictions = run_random_forest(trn_data, tst_data)
    lr_predictions = run_logistic_regression(trn_data, tst_data)
    gbt_predictions = run_gradient_boosted_trees(trn_data, tst_data)

    results = DataFrame(
        rf_predictions = rf_predictions,
        lr_predictions = lr_predictions,
        gbt_predictions = gbt_predictions
    )
    
    return results
end


end

module SurvivalPrediction

using CSV
using DataFrames
using Random
using Statistics

include("../src/Utils.jl")
import .Utils

# import models:
include("RandomForest.jl")
include("LogReg.jl")
include("GradBoost.jl")
import .RandomForest
import .LogReg
import .GradBoost

function run_randforest(X_trn::Matrix, y_trn::Vector, tst_data::Matrix)
    model = RandomForest.train(X_trn, y_trn)
    predictions = RandomForest.predict(model, tst_data)
    return predictions
end

# Placeholder function to run Logistic Regression model (replace with actual code)
function run_logreg(X_trn::Matrix, y_trn::Vector, tst_data::Matrix)
    model = LogReg.train(X_trn, y_trn)
    predictions = LogReg.predict(model, tst_data)
    return predictions
end

# Placeholder function to run Gradient Boosting model (replace with actual code)
function run_gradboost(X_trn::Matrix, y_trn::Vector, tst_data::Matrix)
    model = GradBoost.train(X_trn, y_trn)
    predictions = GradBoost.predict(model, tst_data)
    return predictions
end


function run_models(file_path::String)
    path = joinpath(@__DIR__, file_path) |> normpath
    df = Utils.load_csv(path)
    X_trn, y_trn, X_tst, y_tst = Utils.process_and_split_data(df)

    rf_predictions = run_random_forest(X_trn, y_trn, X_tst)
    lr_predictions = run_logistic_regression(X_trn, y_trn, X_tst)
    gbt_predictions = run_gradient_boosted_trees(X_trn, y_trn, X_tst)

    results = DataFrame(
        rf_predictions = rf_predictions,
        lr_predictions = lr_predictions,
        gbt_predictions = gbt_predictions
    )
    
    return results
end


end

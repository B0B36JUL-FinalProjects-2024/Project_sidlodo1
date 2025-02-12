include("../src/SurvivalPrediction.jl")
using .SurvivalPrediction
SP = SurvivalPrediction
# prepare data

path = joinpath(@__DIR__, "../data/train.csv") |> normpath
X_trn, y_trn, X_tst, y_tst = SP.Utils.load_data_and_split(path; test_ratio=0.2)

# run models
model_LR = SP.LR.LogRegModel(n_iters=100)

method_grad = SP.LR.GradientDescentMethod()
pred = SP.get_prediction(model_LR, method_grad, X_trn, y_trn, X_tst)
accuracy = SP.Utils.classify_predictions(pred, y_tst)

method_new = SP.LR.NewtonMethod()
SP.report_classification(model_LR, method_new, X_trn, y_trn, X_tst, y_tst)

model_RF = SP.RF.RandomForestModel()
SP.report_classification(model_RF, X_trn, y_trn, X_tst, y_tst)

model_GB = SP.GB.GradBoostModel()
pred = SP.get_prediction(model_GB, X_trn, y_trn, X_tst)
accuracy = SP.Utils.classify_predictions(pred, y_tst)

# run models and plot the results

param_sets = [
    Dict(:lr => 0.001, :n_iters => 20),
    Dict(:lr => 0.1, :n_iters => 20)
]

SP.plot_logreg_acc(X_trn, y_trn, X_tst, y_tst, param_sets)

param_sets = [
    Dict(:n_trees => 20, :depth => 5, :features => 3),
    Dict(:n_trees => 20, :depth => 10, :features => 6)
]

SP.plot_trees(X_trn, y_trn, X_tst, y_tst, param_sets)

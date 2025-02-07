include("../src/Utils.jl")
using .Utils

# prepare data

path = joinpath(@__DIR__, "../data/train.csv") |> normpath
df = Utils.load_csv(path)
X_trn, y_trn, X_tst, y_tst = Utils.process_and_split_data(df; test_ratio=0.2)

include("../src/SurvivalPrediction.jl")
using .SurvivalPrediction
SP = SurvivalPrediction

# run models
model_LR = SP.LR.LogRegModel(n_iters=100)

method_grad = SP.LR.GradientDescentMethod(0.01)
pred = SP.get_prediction(model_LR, method_grad, X_trn, y_trn, X_tst)
accuracy = Utils.classify_predictions(pred, y_tst)

method_new = SP.LR.NewtonMethod()
pred = SP.get_prediction(model_LR, method_new, X_trn, y_trn, X_tst)
accuracy = Utils.classify_predictions(pred, y_tst)

model_RF = SP.RF.RandomForestModel()
pred = SP.get_prediction(model_RF, X_trn, y_trn, X_tst)
accuracy = Utils.classify_predictions(pred, y_tst)

model_GB = SP.GB.GradBoostModel()
pred = SP.get_prediction(model_GB, X_trn, y_trn, X_tst)
accuracy = Utils.classify_predictions(pred, y_tst)

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


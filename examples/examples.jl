include("../src/Utils.jl")
import .Utils

# prepare data

path = joinpath(@__DIR__, "../data/train.csv") |> normpath
df = Utils.load_csv(path)
X_trn, y_trn, X_tst, y_tst = Utils.process_and_split_data(df; test_ratio=0.2)

include("../src/SurvivalPrediction.jl")
using .SurvivalPrediction

# run models
pred = SurvivalPrediction.run_logreg(X_trn, y_trn, X_tst; lr=0.01, n_iters=100, method=:grad_descent)
accuracy = Utils.classify_predictions(pred, y_tst)

pred = SurvivalPrediction.run_logreg(X_trn, y_trn, X_tst; lr=0.01, n_iters=100, method=:newton)
accuracy = Utils.classify_predictions(pred, y_tst)

pred = SurvivalPrediction.run_randforest(X_trn, y_trn, X_tst; n_trees=10, max_depth=5, max_features=3)
accuracy = Utils.classify_predictions(pred, y_tst)

pred = SurvivalPrediction.run_gradboost(X_trn, y_trn, X_tst; n_trees=10, max_depth=5, max_features=3)
accuracy = Utils.classify_predictions(pred, y_tst)

# run models and plot the results

param_sets = [
    Dict(:lr => 0.001, :n_iters => 20),
    Dict(:lr => 0.1, :n_iters => 20)
]

SurvivalPrediction.plot_logreg_acc(X_trn, y_trn, X_tst, y_tst, param_sets)

param_sets = [
    Dict(:n_trees => 20, :depth => 5, :features => 3),
    Dict(:n_trees => 20, :depth => 10, :features => 6)
]

SurvivalPrediction.plot_trees(X_trn, y_trn, X_tst, y_tst, param_sets)


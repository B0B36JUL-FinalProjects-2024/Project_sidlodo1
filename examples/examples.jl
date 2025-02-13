using SurvivalPrediction
# prepare data

path = joinpath(@__DIR__, "../data/train.csv") |> normpath

X_trn, y_trn, X_tst, y_tst = load_data_and_split(path; test_ratio=0.2)

# run models
model_LR = LogRegModel(n_iters=100)

method_grad = GradientDescentMethod()
pred = get_prediction(model_LR, method_grad, X_trn, y_trn, X_tst)
accuracy = classify_predictions(pred, y_tst)

method_new = NewtonMethod()
report_classification(model_LR, method_new, X_trn, y_trn, X_tst, y_tst)

model_RF = RandomForestModel()
report_classification(model_RF, X_trn, y_trn, X_tst, y_tst)

model_GB = GradBoostModel()
pred = get_prediction(model_GB, X_trn, y_trn, X_tst)
accuracy = classify_predictions(pred, y_tst)

# run models and plot the results

param_sets = [
    Dict(:lr => 0.001, :n_iters => 20),
    Dict(:lr => 0.1, :n_iters => 20)
]

plot_logreg_acc(X_trn, y_trn, X_tst, y_tst, param_sets)

param_sets = [
    Dict(:n_trees => 20, :depth => 5, :features => 3),
    Dict(:n_trees => 20, :depth => 10, :features => 6)
]

plot_trees(X_trn, y_trn, X_tst, y_tst, param_sets)

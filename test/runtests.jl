module test_SurvivalPrediction

using Test
using Impute
include("../src/Utils.jl")
import .Utils


path = joinpath(@__DIR__, "../data/train.csv") |> normpath
df = Utils.load_csv(path)
X_trn, y_trn, X_tst, y_tst = Utils.process_and_split_data(df)

include("../src/RandomForest.jl")
import .RandomForest

model = RandomForest.train(X_trn, y_trn; n_trees=100, max_depth=5,k=2)
predictions = RandomForest.predict(model, X_tst)
# after 100 trees, the accuracy should be around 0.75 (it doesn't increase)

accuracy = Utils.classify_predictions(predictions, y_tst)



end


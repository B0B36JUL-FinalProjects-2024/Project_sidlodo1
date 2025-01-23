module test_SurvivalPrediction

using Test
include("../src/Utils.jl")
import .Utils


path = joinpath(@__DIR__, "../data/train.csv") |> normpath
df = Utils.load_csv(path)
X_trn, y_trn, X_tst, y_tst = Utils.process_and_split_data(df; test_ratio=0.1)

include("../src/RandomForest.jl")
import .RandomForest

model = RandomForest.train(X_trn, y_trn; n_trees=100, max_depth=5,k=2)
predictions = RandomForest.predict(model, X_tst)
# after 100 trees, the accuracy should be around 0.75 (it doesn't increase)


include("../src/LogReg.jl")
import .LogReg

θ = LogReg.train(X_trn, y_trn; lr=0.2, n_iters=2000, λ=0.2)
predictions = LogReg.predict(θ, X_tst; threshold=0.5)
accuracy = Utils.classify_predictions(predictions, y_tst)



end


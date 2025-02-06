include("../src/Utils.jl")
using .Utils
path = joinpath(@__DIR__, "../data/train.csv") |> normpath
df = Utils.load_csv(path)
X_trn, y_trn, X_tst, y_tst = Utils.process_and_split_data(df; test_ratio=0.2)


include("../src/ClassifierModels.jl")
include("../src/LogReg.jl")
using .ClassifierModels
using .LogReg

model = LogReg.LogRegModel()
method = LogReg.GradientDescentMethod(0.01)
typeof(model)
w, b = LogReg.train(X_trn, y_trn, model, method)

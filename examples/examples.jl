include("../src/SurvivalPrediction.jl")
using .SurvivalPrediction

trn_data_path = "/Users/domisidlova/Downloads/data/train.csv"
tst_data_path = "/Users/domisidlova/Downloads/data/test.csv"

trn_data = SurvivalPrediction.process_data(trn_data_path)
tst_data = SurvivalPrediction.process_data(tst_data_path)

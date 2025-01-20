module test_SurvivalPrediction

using Test
using CSV
using DataFrames
using Impute
using CategoricalArrays

include("../src/Utils.jl")
import .Utils

# trn_path = joinpath(@__DIR__, "../data/train.csv") |> normpath
# trn_df = Utils.process_data(trn_path)
# trn_missing_df = Utils.detect_missing_values(trn_df)
# trn_cleaned = Utils.handle_missing_values(trn_df, :Embarked)
# trn_handled_missing_df = Utils.detect_missing_values(trn_cleaned)

tst_path = joinpath(@__DIR__, "../data/test.csv") |> normpath
tst_df = Utils.process_data(tst_path)
tst_missing_df = Utils.detect_missing_values(tst_df)
tst_cleaned = Utils.handle_missing_values(tst_df, :Fare, :Cabin, :Age)
tst_handled_missing_df = Utils.detect_missing_values(tst_cleaned)

# df = DataFrame(A = [1.0, missing, 3.0, 4.0, missing, 4.0], B = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
# df_copy = Utils.knn_impute(df, :A)


end

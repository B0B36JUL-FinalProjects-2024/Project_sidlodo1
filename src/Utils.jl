module Utils

export load_data_and_split, classify_predictions

using CSV
using DataFrames
using Statistics
using Impute
using CategoricalArrays
using Random

include("../src/NewFeatures.jl")
import .NewFeatures

function load_data_and_split(path::String; test_ratio::Float64=0.2)
    df = load_csv(path)
    X_train, y_train, X_test, y_test = process_and_split_data(df; test_ratio=test_ratio)
    return X_train, y_train, X_test, y_test
end

function load_csv(data_path::String)
    df = CSV.File(data_path) |> DataFrame
    return df
end

function encode(df::DataFrame)
    cols_to_encode = [:Fare, :Embarked, :Title]
    for col in cols_to_encode
        df[!, col] = categorical(df[!, col])
    end
    for col in cols_to_encode
        df[!, col] = levelcode.(df[!, col])
    end
    return df
end

function process_and_split_data(df::DataFrame; test_ratio::Float64=0.2)
    df = process_data(df)

    X = Matrix(df[:, [:Pclass, :Sex, :Age, :SibSp, :Parch, :Fare, :Embarked, :Title]])
    y = df.Survived

    # Split the data into training and test sets
    n_samples = size(X, 1)
    test_size = Int(floor(test_ratio * n_samples))
    indices = shuffle(1:n_samples)
    test_indices = indices[1:test_size]
    train_indices = indices[test_size+1:end]

    X_train = X[train_indices, :]
    y_train = y[train_indices]
    X_test = X[test_indices, :]
    y_test = y[test_indices]

    return X_train, y_train, X_test, y_test
end

function process_data(df::DataFrame)
    df.Sex .= (df.Sex .== "male") .+ 0
    if :Ticket in names(df)
        df = delete_column(df, :Ticket)
    end

    df_cleaned = handle_missing_values(df, :Embarked, :Cabin, :Age)
    df_cleaned = NewFeatures.add_new_features(df_cleaned)
    df_encoded = encode(df_cleaned)
    return df_encoded
end

function delete_rows(df::DataFrame, col::Symbol)
    return dropmissing(df, col)
end

function delete_column(df::DataFrame, col::Symbol)
    select!(df, Not(col))
    return df
end

function knn_impute(df::DataFrame, col::Symbol; k::Int = 5)
    target_col = df[!, col]
    target_col_matrix = reshape(convert(Vector{Union{Missing, Float64}}, target_col), :, 1)
    
    imputed_col = Impute.knn(target_col_matrix, k=k)
    df[!, col] .= imputed_col
    
    return df
end

function handle_missing_values(df::DataFrame, delete_row::Symbol, delete_col::Symbol, impute_col::Symbol)
    new_df = delete_rows(df, delete_row)
    new_df = delete_column(new_df, delete_col)
    new_df = knn_impute(new_df, impute_col)

    return new_df
    
end

function detect_missing_values(df::DataFrame)
    mis_val = [sum(ismissing, df[!, col]) for col in names(df)]
    
    mis_val_percent = [100 * mean(ismissing, df[!, col]) for col in names(df)]
    
    mis_val_table = DataFrame(
        Column = names(df),
        MissingValues = mis_val,
        PercentMissing = mis_val_percent
    )
    
    mis_val_table = filter(row -> row.PercentMissing > 0, mis_val_table)
    
    sort!(mis_val_table, :PercentMissing, rev = true)
    
    println("Your selected dataframe has $(size(df, 2)) columns.")
    
    return mis_val_table
end

function classify_predictions(predictions::AbstractVector, actual::Vector)
    correct = sum(predictions .== actual)
    accuracy = correct / length(actual)
    return accuracy
end

end

module Utils

using CSV
using DataFrames
using Statistics
using Impute

function process_data(trn_data_path::String)
    data = CSV.File(trn_data_path) |> DataFrame

    data.Sex .= (data.Sex .== "male") .+ 0

    return data
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
    
    # Percentage of missing values
    mis_val_percent = [100 * mean(ismissing, df[!, col]) for col in names(df)]
    
    # Create a DataFrame with the results
    mis_val_table = DataFrame(
        Column = names(df),
        MissingValues = mis_val,
        PercentMissing = mis_val_percent
    )
    
    # Filter out columns with 0% missing values
    mis_val_table = filter(row -> row.PercentMissing > 0, mis_val_table)
    
    # Sort by "% of Total Values" in descending order
    sort!(mis_val_table, :PercentMissing, rev = true)
    
    # Print summary information
    println("Your selected dataframe has $(size(df, 2)) columns.")
    println("There are $(size(mis_val_table, 1)) columns that have missing values.")
    
    # Return the dataframe with missing information
    return mis_val_table
end

end

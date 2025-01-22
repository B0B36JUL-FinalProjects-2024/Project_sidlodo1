module test_Utils

using Test
using CSV
using DataFrames
using Impute

include("../src/Utils.jl")
import .Utils

@testset "Utils Tests" begin

    @testset "process_data" begin
        df = DataFrame(Sex = ["male", "female", "male"], Ticket = [1, 2, 3])
        processed_df = Utils.process_data(df)
        @test processed_df.Sex == [1, 0, 1]
    end

    @testset "delete_rows" begin
        df = DataFrame(A = [1, missing, 3], B = [4, 5, 6])
        cleaned_df = Utils.delete_rows(df, :A)
        @test size(cleaned_df, 1) == 2
    end

    @testset "delete_column" begin
        df = DataFrame(A = [1, 2, 3], B = [4, 5, 6])
        cleaned_df = Utils.delete_column(df, :B)
        @test size(cleaned_df, 2) == 1
        @test "B" ∉ names(cleaned_df)
    end

    @testset "knn_impute" begin
        df = DataFrame(A = [1.0, missing, 3.0, 4.0, missing, 4.0], B = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        imputed_df = Utils.knn_impute(df, :A)
        @test !any(ismissing, imputed_df.A)
    end

    @testset "handle_missing_values" begin
        df = DataFrame(A = [1.0, missing, 3.0, 4.0, missing, 4.0, 6.0], B = [2.0, 3.0, 4.0, 5.0, missing, 7.0, 7.0], C = [missing, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0])
        cleaned_df = Utils.handle_missing_values(df, :B, :A, :C)
        @test size(cleaned_df, 2) == 2
        @test !any(ismissing, cleaned_df.C)
    end

    @testset "detect_missing_values" begin
        df = DataFrame(A = [1.0, missing, 3.0], B = [4.0, 5.0, missing])
        missing_df = Utils.detect_missing_values(df)
        @test size(missing_df, 1) == 2
        @test missing_df.PercentMissing[1] == 33.33333333333333
    end

    @testset "classify_predictions" begin
        predictions = [0, 0, 0, 1]
        accuracy = Utils.classify_predictions(predictions, [0, 0, 1, 1])
        @test accuracy == 0.75
    end
    
end

end

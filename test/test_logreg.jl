module test_LogReg

using Test

include("../src/ClassifierModels.jl")
include("../src/LogReg.jl")
using .LogReg

@testset "LogReg Tests" begin
    X = [1 5;
        2 6;
        1 4;
        1 6;
        3 4;
        2 4]
    y = [0, 1, 0, 1, 0, 1]
    n, m = size(X)
    X_norm = LogReg.normalize(X)
    w = zeros(m)

    @testset "sigmoid" begin
        @test LogReg.sigmoid(0) == 0.5
        @test isapprox(LogReg.sigmoid(-1), 0.2689, atol=1e-4)
        @test isapprox(LogReg.sigmoid(1), 0.7310, atol=1e-4)
    end

    @testset "grad_descent" begin
        method = LogReg.GradientDescentMethod()
        w_new, b_new = LogReg.update!(X_norm, y, w, 0.0, method)
        @test isapprox(w_new, [0.0000, 0.0025], atol=1e-4)
        @test b_new == 0.0
    end

    @testset "newton" begin
        method = LogReg.NewtonMethod()
        w_new = LogReg.update!(X, y, w, method)
        @test isapprox(w_new, [-0.4081, 0.1736], atol=1e-4)
    end

    @testset "train_grad" begin
        model = LogReg.LogRegModel(;n_iters=3)
        method = LogReg.GradientDescentMethod()
        w_new, b_new = LogReg.train(X, y, model, method)
        @test isapprox(w_new, [0.0000, 0.0076], atol=1e-4)
        @test isapprox(b_new, 0.0000, atol=1e-4)
    end
    @testset "train_new" begin
        model = LogReg.LogRegModel(;n_iters=3)
        method = LogReg.NewtonMethod()
        w_new = LogReg.train(X, y, model, method)
        @test isapprox(w_new, [0.6918, 1.8045], atol=1e-4)
    end

    @testset "predict" begin
        model = LogReg.LogRegModel(;n_iters=3)
        method = LogReg.GradientDescentMethod()
        w_new, b_new = LogReg.train(X, y, model, method)
        predictions = LogReg.predict(w_new, b_new, X)
        @test predictions == [1, 1, 0, 1, 0, 0]
    end

end

end

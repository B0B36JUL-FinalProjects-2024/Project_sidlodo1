module ClassifierModels

export GradientDescentMethod, NewtonMethod, LogRegModel, EnsembleMethod, RandomForestMethod, GradBoostMethod, Classifier, LogRegOptimizationMethod

abstract type Classifier end

abstract type LogRegOptimizationMethod end

struct GradientDescentMethod <: LogRegOptimizationMethod
    lr::Float64
end

struct NewtonMethod <: LogRegOptimizationMethod
end

struct LogRegModel <: Classifier
    n_iters::Int
    function LogRegModel(;n_iters=20)
        new(n_iters)
    end
end

abstract type EnsembleMethod end

struct RandomForestMethod <: EnsembleMethod end
struct GradBoostMethod <: EnsembleMethod end

end

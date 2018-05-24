using DataFrames
include("type.jl")


function predict(result::ClusteringResult, target::DataFrame)
    predictedValues = _predict(result, target)
    return predictedValues
end

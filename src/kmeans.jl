using DataFrames
using StatsBase
include("dist.jl")

type kMeansResults
    x::DataFrames.DataFrame
    k::Int
    estimatedClass::Array{Int}
    iterCount::Int
end

function kMeans(data, k)
    dataPointsNum = size(data, 1)
    estimatedClass = Array{Int}(dataPointsNum)
    sample!(1:k, estimatedClass)

    iterCount = 0
    while true
        # update representative points
        representativePoints = []
        for representativeIndex in 1:k
            groupIndex = find(estimatedClass .== representativeIndex)
            groupData = data[groupIndex, :]

            # TODO: check the return type of colwise
            representativePoint = [ valArray[1] for valArray in colwise(mean, groupData) ]
            push!(representativePoints, representativePoint)
        end

        # update group belonging
        tempEstimatedClass = Array{Int}(dataPointsNum)
        for dataIndex in 1:dataPointsNum
            dataPoint = Array(data[dataIndex, :])
            distances = Array{Float64}(k)
            for representativeIndex in 1:k
                distances[representativeIndex] = calcDist(dataPoint, representativePoints[representativeIndex])
            end

            # TODO: check the existence of argmin
            tempEstimatedClass[dataIndex] = sortperm(distances)[1]
        end

        if estimatedClass == tempEstimatedClass
            iterCount += 1
            break
        end
        estimatedClass = tempEstimatedClass
        iterCount += 1
    end
    return kMeansResults(data, k, estimatedClass, iterCount)
end

function calcDist(sourcePoint::Array, destPoint::Array; method="euclidean")

    if length(sourcePoint) != length(destPoint)
        error("The lengths of two arrays are different.")
        return
    end

    if method == "euclidean"
        return euclidean(sourcePoint, destPoint)
    elseif method == "minkowski"
        return minkowski(sourcePoint, destPoint)
    end
end

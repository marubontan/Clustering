using DataFrames
using StatsBase
include("dist.jl")

type kMeans
    x::DataFrames.DataFrame
    k::Int
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

function fit(kMeans::kMeans)
    dataPointsNum = size(kMeans.x, 1)
    estimatedClass = Array{Int}(dataPointsNum)
    sample!(1:kMeans.k, estimatedClass)

    while true
        # update representative points
        representativePoints = []
        for representativeIndex in 1:kMeans.k
            groupIndex = find(estimatedClass .== representativeIndex)
            groupData = kMeans.x[groupIndex, :]

            # TODO: check the return type of colwise
            representativePoint = [ valArray[1] for valArray in colwise(mean, groupData) ]
            push!(representativePoints, representativePoint)
        end

        # update group belonging
        tempEstimatedClass = Array{Int}(dataPointsNum)
        for dataIndex in 1:dataPointsNum
            dataPoint = Array(kMeans.x[dataIndex, :])
            distances = Array{Float64}(kMeans.k)
            for representativeIndex in 1:kMeans.k
                distances[representativeIndex] = calcDist(dataPoint, representativePoints[representativeIndex])
            end

            # TODO: check the existence of argmin
            tempEstimatedClass[dataIndex] = sortperm(distances)[1]
        end

        if estimatedClass == tempEstimatedClass
            break
        end
        estimatedClass = tempEstimatedClass
    end
    return estimatedClass
end

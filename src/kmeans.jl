using DataFrames
using StatsBase
include("dist.jl")
include("utils.jl")

type kMeansResults
    x::DataFrames.DataFrame
    k::Int
    estimatedClass::Array{Int}
    iterCount::Int
    costArray::Array{Float64}
end

"""
    kMeans(data, k)

Do clustering to data with K-means algorithm.

# Arguments
- `data::DataFrame`: the clustering target data.
- `k::Int`: the number of clusters.

# Examples
```julia-kMeans
julia> kMeans(DataFrame(x = [1,2,3], y = [4,5,6]),2)
kMeansResults(3×2 DataFrames.DataFrame
│ Row │ x │ y │
├─────┼───┼───┤
│ 1   │ 1 │ 4 │
│ 2   │ 2 │ 5 │
│ 3   │ 3 │ 6 │, 2, [1, 1, 1], 1, [4.0])
```
"""
function kMeans(data::DataFrame, k::Int)

    # initialize
    dataPointsNum = size(data, 1)
    estimatedClass = assignRandomKClass(dataPointsNum, k)

    iterCount = 0
    costArray = Float64[]
    while true

        # update
        representativePoints = updateRepresentative(data, estimatedClass, k)
        tempEstimatedClass, cost = updateGroupBelonging(data, dataPointsNum, representativePoints, k)

        push!(costArray, cost)

        if judgeConvergence(estimatedClass, tempEstimatedClass)
            iterCount += 1
            break
        end

        estimatedClass = tempEstimatedClass
        iterCount += 1
    end
    return kMeansResults(data, k, estimatedClass, iterCount, costArray)
end

function assignRandomKClass(dataPointsNum, k)
    estimatedClass = Array{Int}(dataPointsNum)
    sample!(1:k, estimatedClass)
    return estimatedClass
end

function updateRepresentative(data::DataFrame, estimatedClass::Array{Int}, k::Int)
    representativePoints = Array{Array{Float64,1}}(k)
    for representativeIndex in 1:k
        groupIndex = find(estimatedClass .== representativeIndex)
        groupData = data[groupIndex, :]

        representativePoint = [ valArray[1] for valArray in colwise(mean, groupData) ]
        representativePoints[representativeIndex] = representativePoint
    end
    return representativePoints
end

function updateGroupBelonging(data::DataFrame, dataPointsNum::Int, representativePoints::Array{Array{Float64, 1}}, k::Int)
    tempEstimatedClass = Array{Int}(dataPointsNum)

    cost = 0.0
    for dataIndex in 1:dataPointsNum
        dataPoint = Array(data[dataIndex, :])
        distances = Array{Float64}(k)
        for representativeIndex in 1:k
            distances[representativeIndex] = calcDist(dataPoint, representativePoints[representativeIndex])
        end

        # TODO: check the existence of argmin
        # TODO: this cost calculation is bad hack
        classIndex = returnArgumentMin(distances)
        tempEstimatedClass[dataIndex] = classIndex
        cost += distances[classIndex] ^ 2
    end
    return tempEstimatedClass, cost
end

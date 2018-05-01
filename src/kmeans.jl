using DataFrames
using StatsBase
include("dist.jl")
include("utils.jl")

struct KMeansResults
    x::DataFrames.DataFrame
    k::Int
    estimatedClass::Array{Int}
    centroids::Array{Array}
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
KMeansResults(3×2 DataFrames.DataFrame
│ Row │ x │ y │
├─────┼───┼───┤
│ 1   │ 1 │ 4 │
│ 2   │ 2 │ 5 │
│ 3   │ 3 │ 6 │, 2, [1, 1, 1], 1, [4.0])
```
"""
function kMeans(data::DataFrame, k::Int; initializer=nothing)

    # initialize
    dataPointsNum = size(data, 1)
    estimatedClass = assignRandomKClass(dataPointsNum, k)
    if initializer == "kmeans++"
        centroids = kMeansPlusPlus(data, k)
    else
        centroids = updateCentroids(data, estimatedClass, k)
    end

    iterCount = 0
    centroidsArray = []
    costArray = Float64[]
    while true

        # update
        tempEstimatedClass, cost, nearestDist = updateGroupBelonging(data, dataPointsNum, centroids, k)

        push!(costArray, cost)

        centroids = updateCentroids(data, tempEstimatedClass, k)

        if length(Set(tempEstimatedClass)) < k
            centroids = assignDataOnEmptyCluster(data, tempEstimatedClass, centroids, nearestDist)
        end

        push!(centroidsArray, centroids)

        if judgeConvergence(estimatedClass, tempEstimatedClass)
            iterCount += 1
            break
        end

        estimatedClass = tempEstimatedClass
        iterCount += 1
    end
    return KMeansResults(data, k, estimatedClass, centroidsArray, iterCount, costArray)
end

function assignRandomKClass(dataPointsNum, k)
    estimatedClass = Array{Int}(dataPointsNum)
    sample!(1:k, estimatedClass)
    return estimatedClass
end

function updateCentroids(data::DataFrame, estimatedClass::Array{Int}, k::Int)
    centroids = Array{Array{Float64,1}}(k)
    for centroidIndex in 1:k
        groupIndex = find(estimatedClass .== centroidIndex)
        groupData = data[groupIndex, :]

        centroid = [ valArray[1] for valArray in colwise(mean, groupData) ]
        centroids[centroidIndex] = centroid
    end
    return centroids
end

function updateGroupBelonging(data::DataFrame, dataPointsNum::Int, centroids::Array, k::Int)
    tempEstimatedClass = Array{Int}(dataPointsNum)

    cost = 0.0
    distanceBetweenDataPointAndNearestCentroid = []
    for dataIndex in 1:dataPointsNum
        dataPoint = Array(data[dataIndex, :])
        distances = Array{Float64}(k)
        for centroidIndex in 1:k
            distances[centroidIndex] = calcDist(dataPoint, centroids[centroidIndex])
        end

        push!(distanceBetweenDataPointAndNearestCentroid, minimum(distances))
        classIndex = returnArgumentMin(distances)
        tempEstimatedClass[dataIndex] = classIndex

        # TODO: this cost calculation is bad hack
        cost += distances[classIndex] ^ 2
    end
    return tempEstimatedClass, cost, distanceBetweenDataPointAndNearestCentroid
end

function assignDataOnEmptyCluster(data::DataFrame, label, centers, nearestDist)
    emptyCluster = findEmptyCluster(label, centers)
    nearestDistProb = makeValuesProbabilistic(nearestDist)
    pickedDataPointsIndex = stochasticallyPickUp(Array(1:nrow(data)), nearestDistProb, length(emptyCluster))

    for (i,cluster) in enumerate(emptyCluster)
        centers[cluster] = vec(Array(data[pickedDataPointsIndex[i], :]))
    end
    return centers
end

function findEmptyCluster(label, centers)
    emptyCluster = collect(setdiff(Set(1:length(centers)), Set(label)))
    return emptyCluster
end

function makeValuesProbabilistic(values)
    return values / sum(values)
end

function stochasticallyPickUp(values, probs, n)
    indexProb = Dict()
    for key in 1:length(values)
        indexProb[key] = probs[key]
    end

    pickedValues = []
    for _ in 1:n
        border = rand(1)[1]

        sum = 0
        for pair in indexProb
            sum += pair[2]
            if sum > border
                push!(pickedValues, pair[1])

                # TODO: it's bad hack to delte the loop target in the loop
                delete!(indexProb, pair[1])

                denominator = 1 - pair[2]
                for (key,val) in indexProb
                    indexProb[key] = val / denominator
                end
                break
            end
        end
    end
    return pickedValues
end

function kMeansPlusPlus(data::DataFrame, k::Int)
    dataDict = dataFrameToDict(data)
    ind, centroid = randomlyChooseOneDataPoint(dataDict)
    distanceDict = calcDistBetweenCenterAndDataPoints(dataDict, ind)
    distanceProbDict = makeDictValueProbabilistic(distanceDict)
    centroidsIndex = wrapperToStochasticallyPickUp(distanceProbDict, k)
    centroids = dataFrame2JaggedArray(data[Int.(centroidsIndex), :])
    return centroids
end

function randomlyChooseOneDataPoint(data::Dict)
    randomIndex = rand([k for k in keys(data)], 1)[1]
    return randomIndex, data[randomIndex]
end

function dataFrameToDict(data::DataFrame)

    indexDataDict = Dict()
    for i in 1:nrow(data)
        indexDataDict[i] = vec(Array(data[i, :]))
    end

    return indexDataDict
end

function calcDistBetweenCenterAndDataPoints(data::Dict, index::Int)
    center = data[index]
    distanceDict = Dict()
    for pair in data
        if pair[1] == index
            continue
        end

        distanceDict[pair[1]] = calcDist(center, pair[2])
    end
    return distanceDict
end

function makeDictValueProbabilistic(data)
    vals = [v for v in values(data)]
    valsSum = sum(vals)

    for pair in data
        data[pair[1]] = pair[2] / valsSum
    end

    return data
end

function wrapperToStochasticallyPickUp(data::Dict, n::Int)
    index = []
    probs = []
    for pair in data
        push!(index, pair[1])
        push!(probs, pair[2])
    end

    return stochasticallyPickUp(index, probs, n)
end

function dataFrame2JaggedArray(data::DataFrame)
    returnArray = []
    for i in 1:nrow(data)
        push!(returnArray, vec(Array(data[i,:])))
    end
    return returnArray
end


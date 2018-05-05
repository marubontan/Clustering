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
        classIndex = indmin(distances)
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

function kMeansPlusPlus(data::DataFrame, k::Int)
    dataDict = dataFrameToDict(data)
    ind = randomlyChooseOneDataPoint(dataDict)
    distanceDict = calcDistBetweenCenterAndDataPoints(dataDict, ind)

    distanceProbDict = makeDictValueProbabilistic(distanceDict)

    centroidsIndices = [ind]
    for i in 1:k-1
        centroidsIndex = wrapperToStochasticallyPickUp(distanceProbDict, 1)[1]
        push!(centroidsIndices, centroidsIndex)

        if i == k-1
            break
        end

        distanceDict = updateDistanceDict(distanceDict, dataDict, centroidsIndex)

        distanceProbDict = makeDictValueProbabilistic(distanceDict)
    end

    centroids = dataFrame2JaggedArray(data[Int.(centroidsIndices), :])
    return centroids
end

function randomlyChooseOneDataPoint(data::Dict)
    randomIndex = rand([k for k in keys(data)], 1)[1]
    return randomIndex
end

function wrapperToStochasticallyPickUp(data::Dict, n::Int)
    index = []
    probs = Float64[]
    for pair in data
        push!(index, pair[1])
        push!(probs, pair[2])
    end

    return stochasticallyPickUp(index, probs, n)
end

function updateDistanceDict(distanceDict, dataDict, ind)
    distanceBetweenNewCentroidAndDataPoints = calcDistBetweenCenterAndDataPoints(dataDict, ind)
    for pair in distanceDict
        if pair[2] > distanceBetweenNewCentroidAndDataPoints[pair[1]]
            distanceDict[pair[1]] = distanceBetweenNewCentroidAndDataPoints[pair[1]]
        end
    end
    return distanceDict
end


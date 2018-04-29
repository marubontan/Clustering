using DataFrames
using StatsBase
include("dist.jl")
include("utils.jl")

struct kMeansResults
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
    centroidsArray = []
    centroids = updateCentroids(data, estimatedClass, k)
    costArray = Float64[]
    while true

        # update
        tempEstimatedClass, cost, nearestDist = updateGroupBelonging(data, dataPointsNum, centroids, k)

        push!(costArray, cost)

        centroids = updateCentroids(data, estimatedClass, k)

        if length(Set(tempEstimatedClass)) < k
            centroids = assignDataOnEmptyCluster(data, estimatedClass, centroids, nearestDist)
        end

        push!(centroidsArray, centroids)

        if judgeConvergence(estimatedClass, tempEstimatedClass)
            iterCount += 1
            break
        end

        estimatedClass = tempEstimatedClass
        iterCount += 1
    end
    return kMeansResults(data, k, estimatedClass, centroidsArray, iterCount, costArray)
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

function updateGroupBelonging(data::DataFrame, dataPointsNum::Int, centroids::Array{Array{Float64, 1}}, k::Int)
    tempEstimatedClass = Array{Int}(dataPointsNum)

    cost = 0.0
    distanceBetweenDataPointAndNearestCentroid = []
    for dataIndex in 1:dataPointsNum
        dataPoint = Array(data[dataIndex, :])
        distances = Array{Float64}(k)
        for centroidIndex in 1:k
            distances[centroidIndex] = calcDist(dataPoint, centroids[centroidIndex])
        end

        # TODO: check the existence of argmin
        # TODO: this cost calculation is bad hack
        push!(distanceBetweenDataPointAndNearestCentroid, minimum(distances))
        classIndex = returnArgumentMin(distances)
        tempEstimatedClass[dataIndex] = classIndex
        cost += distances[classIndex] ^ 2
    end
    return tempEstimatedClass, cost, distanceBetweenDataPointAndNearestCentroid
end

function assignDataOnEmptyCluster(data::DataFrame, label, centers, nearestDist)
    # find empty cluster
    emptyCluster = findEmptyCluster(label, centers)
    # make the distance array probabilistic
    nearestDistProb = makeValuesProbabilistic(nearestDist)
    # stochastically pick up the centroids
    pickedDataPointsIndex = stochasticallyPickUp(Array(1:nrow(data)), nearestDistProb, length(emptyCluster))
    # update the vanished centroids
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

function stochasticallyPickUp(values, probs, k)
    indexProb = Dict()
    for key in 1:length(values)
        indexProb[key] = probs[key]
    end

    newCluster = []
    for i in 1:k
        border = rand(1)[1]

        sum = 0
        for pair in indexProb
            sum += pair[2]
            if sum > border
                push!(newCluster, pair[1])
                delete!(indexProb, pair[1])
                denominator = 1 - pair[2]
                for (key,val) in indexProb
                    indexProb[key] = val / denominator
                end
                break
            end
        end
    end
    return newCluster
end

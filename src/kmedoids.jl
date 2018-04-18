using DataFrames
include("utils.jl")

type kMedoidsResults
    x
    k::Int
    estimatedClass::Array{Int}
    medoids::Array{Array}
    iterCount::Int
end

"""
    kMedoids(distanceMatrix, k)

Do clustering to distanceMatrix with K-medoids algorithm

# Argument
- `distanceMatrix`: the distance matrix of data
- `k::Int`: the number of clusters

# Examples
```julia-kMedoids
julia> import Distances

julia> distanceMatrix = Distances.pairwise(Distances.SqEuclidean(), [1 2 3;4 5 6])
3×3 Array{Int64,2}:
 0  2  8
 2  0  2
 8  2  0

julia> kMedoids(distanceMatrix, 2)
kMedoidsResults([0 2 8; 2 0 2; 8 2 0], 2, [2, 1, 1], Array[[2, 1]], 1)
```
"""
function kMedoids(distanceMatrix, k::Int)
    # initialize
    medoidsIndices = randomlyAssignMedoids(distanceMatrix, k)

    iterCount = 0
    updatedGroupInfo = []
    medoids = []
    while true
        updatedGroupInfo = updateGroupBelonging(distanceMatrix, medoidsIndices)
        updatedMedoids = updateMedoids(distanceMatrix, updatedGroupInfo, k)
        push!(medoids, updatedMedoids)

        if judgeConvergence(medoidsIndices, updatedMedoids)
            iterCount += 1
            break
        end
        medoidsIndices = updatedMedoids
        iterCount += 1
    end
    return kMedoidsResults(distanceMatrix, k, updatedGroupInfo, medoids, iterCount, )
end

function randomlyAssignMedoids(distanceMatrix, k::Int)
    dataPointsNum = size(distanceMatrix)[1]
    return shuffle(1:dataPointsNum)[1:k]
end

function updateMedoids(distanceMatrix, groupInfo, k::Int)
    medoidsIndices = Array{Int}(k)
    for class in 1:k
        classIndex = find(groupInfo .== class)
        classDistanceMatrix = distanceMatrix[classIndex, classIndex]
        distanceSum = vec(sum(classDistanceMatrix, 2))
        medoidIndex = classIndex[returnArgumentMin(distanceSum)]
        medoidsIndices[class] = medoidIndex
    end
    return medoidsIndices
end

function updateGroupBelonging(distanceMatrix, representativeIndices::Array{Int})
    dataRepresentativeDistances = referenceDistanceMatrix(distanceMatrix, representativeIndices)

    updatedGroupInfo = Array{Int}(size(dataRepresentativeDistances)[2])
    for i in 1:size(dataRepresentativeDistances)[2]
        updatedGroupInfo[i] = returnArgumentMin(dataRepresentativeDistances[:, i])
    end
    return updatedGroupInfo
end

function referenceDistanceMatrix(distanceMatrix, representativeIndices::Array{Int})
    return distanceMatrix[representativeIndices, :]
end

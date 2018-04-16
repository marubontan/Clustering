using DataFrames

type kMedoidsResults
    x
    k::Int
    estimatedClass::Array{Int}
    medoids::Array{Array}
    iterCount::Int
end

function kMedoids(distanceMatrix, k)
    # initialize
    medoidsIndices = randomlyAssignMedoids(distanceMatrix, k)

    iterCount = 0
    updatedGroupInfo = []
    medoids = []
    while true
        # update group belongings
        updatedGroupInfo = updateGroupBelonging(distanceMatrix, medoidsIndices)

        # update medoids
        updatedMedoids = updateMedoids(distanceMatrix, updatedGroupInfo, k)
        push!(medoids, updatedMedoids)

        if medoidsIndices == updatedMedoids
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

function returnArgumentMin(targetArray::Array)
    return sortperm(targetArray)[1]
end

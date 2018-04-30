using Base.Test
using Distributions
using DataFrames
import Distances
include("../src/dist.jl")
include("../src/kmeans.jl")
include("../src/kmedoids.jl")

@testset "distance function" begin
    dataSourceNum = [1.0, 2.0]
    dataDestNum = [3.0, 4.0]
    @test euclidean(dataSourceNum, dataDestNum) == sqrt(8)
    @test minkowski(dataSourceNum, dataDestNum) == 4

    dataSourceStr = ["cat", "dog"]
    dataDestStr = ["human", "fish"]
    @test_throws MethodError euclidean(dataSourceStr, dataDestStr)
    @test_throws MethodError minkowski(dataSourceStr, dataDestStr)

    dataSourceLen = [1.0, 2.0]
    dataDestLen = [3.0, 4.0, 5.0]
    @test_throws ErrorException euclidean(dataSourceLen, dataDestLen)
    @test_throws ErrorException minkowski(dataSourceLen, dataDestLen)
end

@testset "K-means test" begin
    function makeData()
        groupOne = rand(MvNormal([10.0, 10.0], 5.0 * eye(2)), 100)
        groupTwo = rand(MvNormal([0.0, 0.0], 10 * eye(2)), 100)
        return hcat(groupOne, groupTwo)'
    end

    data = makeData()
    k = 2

    @testset "assignRandomClass" begin
        randomClass = assignRandomKClass(size(data)[1], k)
        @test randomClass[1] in 1:k
        @test randomClass[2] in 1:k
    end

    @testset "updateCentroids" begin
        randomClass = assignRandomKClass(size(data)[1], k)
        centroids = updateCentroids(DataFrame(data), randomClass, k)
        @test isa(centroids, Array{Array{Float64, 1}})
        @test length(centroids) == k
    end

    @testset "assignDataOnEmptyCluster" begin
        x = [1, 1, 3, 3, 7, 7, 9, 9]
        y = [8, 10, 8, 10, 1, 3, 1, 3]
        dataOne = DataFrame(x=x, y=y)
        labelOne = [1, 1, 1, 1, 2, 2, 2, 2]
        centerOne = [[2, 9], [8, 2], [NaN, NaN]]
        nearestDistOne = [sqrt(2), sqrt(2), sqrt(2), sqrt(2), sqrt(2), sqrt(2), sqrt(2), sqrt(2)]
        updatedCenterOne = assignDataOnEmptyCluster(dataOne, labelOne, centerOne, nearestDistOne)

        @test updatedCenterOne[1:2] == centerOne[1:2]
        @test findEmptyCluster(labelOne, centerOne) == [3]

        arrayOne = [1,2,3,4]
        @test makeValuesProbabilistic(arrayOne) == [0.1, 0.2, 0.3, 0.4]
        probOne = [0.2, 0.4, 0.1, 0.3]
        k = 2
        pickedUp = stochasticallyPickUp(arrayOne, probOne, k)
        @test length(pickedUp) == k
        @test typeof(pickedUp[1]) == Int

    end

    @testset "updateGroupBelonging" begin
        groupInfo, cost, nearestDist = updateGroupBelonging(DataFrame(data), size(data)[1], [[0.0, 0.0], [1.0, 2.0]], k)
        @test isa(groupInfo, Array{Int})
        @test isa(cost, Float64)
        @test length(groupInfo) == size(data)[1]
    end

    @testset "kMeans++" begin
        xA = [1.0, 2.0, 2.0, 3.0]
        yA = [3.0, 2.0, 3.0, 8.0]
        dataA = DataFrame(x=xA, y=yA)
        k = 2
        @test_nowarn kMeansPlusPlus(dataA, k)
        dataADict = Dict(1 => [1.0, 3.0], 2 => [2.0, 2.0], 3 => [2.0, 3.0], 4 => [3.0, 8.0])
        @test dataFrameToDict(dataA) == dataADict
        ind, coordinate = randomlyChooseOneDataPoint(dataADict)
        @test coordinate in [[1.0, 3.0], [2.0, 2.0], [2.0, 3.0], [3.0, 8.0]]

        indA = 2
        @test_nowarn calcDistBetweenCenterAndDataPoints(dataADict, indA)
        @test makeDictValueProbabilistic(Dict(1 => 1.0, 2 => 4.0)) == Dict(1 => 0.2, 2 => 0.8)

        dictA = Dict(1 => 0.2, 2 => 0.4, 3 => 0.1, 4 => 0.3)
        n = 2
        @test_nowarn wrapperToStochasticallyPickUp(dictA, n)
        pickedUp = wrapperToStochasticallyPickUp(dictA, n)
        @test length(pickedUp) == n
        @test typeof(pickedUp[1]) == Int

        dataFrameA = DataFrame(x=[1,2,3], y=[3,4,5])
        @test dataFrame2JaggedArray(dataFrameA) == [[1, 3], [2, 4], [3, 5]]

    end

    results = kMeans(DataFrame(data), k)

    @test isa(results, KMeansResults)
    @test size(results.x) == size(data)
    @test results.k == k
    @test length(results.estimatedClass) ==  size(data)[1]
    @test length(results.centroids) == results.iterCount
    @test length(results.costArray) == results.iterCount
    @test length(Set(results.estimatedClass)) == k
end

@testset "K-medoids test" begin
    function makeData()
        groupOne = rand(MvNormal([10.0, 10.0], 5.0 * eye(2)), 100)
        groupTwo = rand(MvNormal([0.0, 0.0], 10 * eye(2)), 100)
        return hcat(groupOne, groupTwo)'
    end

    data = makeData()'
    k = 2
    distanceMatrix = Distances.pairwise(Distances.SqEuclidean(), data)

    @testset "randomlyAssignMediods" begin
        @test isa(randomlyAssignMedoids(distanceMatrix, 2), Array)
        @test length(randomlyAssignMedoids(distanceMatrix, 2)) == 2
        medoidsIndices = randomlyAssignMedoids(distanceMatrix, 2)
        @test medoidsIndices[1] != medoidsIndices[2]
    end

    @testset "updateMedoids" begin
        updatedMedoids, cost = updateMedoids(distanceMatrix, rand(1:k, size(distanceMatrix)[1]), k)
        @test isa(updatedMedoids, Array{Int})
        @test length(updatedMedoids) == 2
        @test updatedMedoids[1] != updatedMedoids[2]
    end

    @testset "updateGroupBelonging" begin
        @test isa(updateGroupBelonging(distanceMatrix, [5, 10]), Array{Int})
        @test length(updateGroupBelonging(distanceMatrix, [5, 10])) == size(distanceMatrix)[1]
    end

    @test returnArgumentMin([3, 1, 2]) == 2

    results = kMedoids(distanceMatrix, k)

    @test isa(results, kMedoidsResults)
    @test size(results.x) == size(distanceMatrix)
    @test results.k == k
    @test length(results.estimatedClass) ==  size(data)[2]
    @test length(results.medoids) == results.iterCount
    @test length(results.costArray) == results.iterCount
end

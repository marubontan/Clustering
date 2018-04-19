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

    results = kMeans(DataFrame(data), k)

    @test isa(results, kMeansResults)
    @test size(results.x) == size(data)
    @test results.k == k
    @test length(results.estimatedClass) ==  size(data)[1]
    @test length(results.centroids) == results.iterCount
    @test length(results.costArray) == results.iterCount
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

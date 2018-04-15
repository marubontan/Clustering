using Base.Test
using Distributions
using DataFrames
include("../src/dist.jl")
include("../src/kmeans.jl")

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
    results = kMeans(DataFrame(data), k)

    @test isa(results, kMeansResults)
    @test size(results.x) == size(data)
    @test results.k == k
    @test length(results.estimatedClass) ==  size(data)[1]
end

@testtest "K-medoids test" begin
    function makeData()
        groupOne = rand(MvNormal([10.0, 10.0], 5.0 * eye(2)), 100)
        groupTwo = rand(MvNormal([0.0, 0.0], 10 * eye(2)), 100)
        return hcat(groupOne, groupTwo)'
    end

    data = makeData()'
    k = 2
    distanceMatrix = pairwise(SqEuclidean(), data)

    resulsts = kMedoids(distanceMatrix, k)

    @test isa(results, kMedoidsResults)
    @test size(results.x) == size(distanceMatrix)
    @test results.k == k
    @test length(results.estimatedClass) ==  size(data)[1]
end

using Test
using Distributions
using DataFrames
using Distances
using Combinatorics
using MLBase
using Random

include("../src/kmeans.jl")
include("../src/kmedoids.jl")


function makeData()
    eye = zeros(2, 2)
    for i = 1:2
        eye[i, i] = 1.0
    end
    groupOne = rand(MvNormal([10.0, 10.0], 5.0 * eye), 1000)
    groupTwo = rand(MvNormal([0.0, 0.0], 10 * eye), 1000)
    groupThree = rand(MvNormal([100.0, 100.0], 20 * eye), 1000)
    groupFour = rand(MvNormal([100.0, 50.0], 20 * eye), 1000)
    return hcat(groupOne, groupTwo, groupThree, groupFour)'
end

Random.seed!(1234)
trainData = makeData()

timeKmeans = @elapsed resultKmeans = kMeans(DataFrame(trainData), 4)
timeKmeansPlusPlus = @elapsed resultKmeansPlusPlus = kMeans(DataFrame(trainData), 4; initializer="kmeans++")

distanceMatrix = Distances.pairwise(Distances.SqEuclidean(), trainData')
timeKmedoids = @elapsed resultKmedoids = kMedoids(distanceMatrix, 4)

# time
@show timeKmeans
@show timeKmeansPlusPlus
@show timeKmedoids


# accuracy

#TODO: this function costs
function calculateAccuracy(groundTruth, predicted)
  maxCombination = []
  maxAccuracy = 0.0

  for combination in permutations(collect(Set(predicted)))
      conv = Dict()
      sortedGroundTruth = sort(collect(Set(groundTruth)))
      for pair in zip(sortedGroundTruth, combination)
          conv[pair[1]] = pair[2]
      end
      target = Array{Int}(undef, length(groundTruth))
      for (i,label) in enumerate(groundTruth)
          target[i] = conv[label]
      end
      accuracy = correctrate(target, predicted)
      if accuracy > maxAccuracy
          maxAccuracy = accuracy
          maxCombination = combination
      end
  end
  return maxAccuracy
end

testData = makeData()
groundTruth = []
for i in range(1, length=4000)
    if 1 <= i <=1000
        push!(groundTruth, 1)
    elseif 1001 <= i <= 2000
        push!(groundTruth, 2)
    elseif 2001 <= i <= 3000
        push!(groundTruth, 3)
    else
        push!(groundTruth, 4)
    end
end


kMeansScore = predict(resultKmeans, DataFrame(testData))
kMeansPlusPlusScore = predict(resultKmeansPlusPlus, DataFrame(testData))

@show calculateAccuracy(groundTruth, kMeansScore)
@show calculateAccuracy(groundTruth, kMeansPlusPlusScore)

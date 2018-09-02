using DataFrames
using Distributions
import Distances
include("../src/kmedoids.jl")

function makeData()
    eye = zeros(2, 2)
    for i = 1:2
        eye[i, i] = 1.0
    end
    groupOne = rand(MvNormal([10.0, 10.0], 5.0 * eye), 100)
    groupTwo = rand(MvNormal([0.0, 0.0], 10 * eye), 100)
    return hcat(groupOne, groupTwo)'
end

function main()
    data = makeData()'

    k = 2
    distanceMatrix = Distances.pairwise(Distances.SqEuclidean(), data)

    results = kMedoids(distanceMatrix, k)
    print(results)
end

main()

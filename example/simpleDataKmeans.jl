using DataFrames
using Distributions
include("../src/kmeans.jl")

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
    data = makeData()

    kmeansResult = kMeans(DataFrame(data), 2; initializer="kmeans++")
    print(kmeansResult)
end

main()

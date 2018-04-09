using DataFrames
using Distributions
using PyPlot
include("../src/kmeans.jl")

function makeData()
    groupOne = rand(MvNormal([10.0, 10.0], 5.0 * eye(2)), 100)
    groupTwo = rand(MvNormal([0.0, 0.0], 10 * eye(2)), 100)
    return hcat(groupOne, groupTwo)'
end

function main()
    data = makeData()

    kmeansResult = kMeans(DataFrame(data), 2)
    print(kmeansResult)
end

main()

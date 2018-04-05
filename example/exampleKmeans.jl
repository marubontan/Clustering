using DataFrames
using Distributions
using PyPlot
include("../src/kmeans.jl")

function splitTrainTest(data, at = 0.7)
    n = nrow(data)
    ind = shuffle(1:n)
    train_ind = view(ind, 1:floor(Int, at*n))
    test_ind = view(ind, (floor(Int, at*n)+1):n)
    return data[train_ind,:], data[test_ind,:]
end

function makeData()
    groupOne = rand(MvNormal([10.0, 10.0], 5.0 * eye(2)), 100)
    groupTwo = rand(MvNormal([0.0, 0.0], 10 * eye(2)), 100)
    return hcat(groupOne, groupTwo)'
end

function main()

    data = makeData()

    knn = kMeans(DataFrame(data), 2)
    predictedClass = fit(knn)
    print(predictedClass)
end

main()

using MLDatasets
using DataFrames
include("../src/kmeans.jl")

train_x, train_y = MNIST.traindata()

sub_x = zeros(100, 784)
for i in 1:100
    sub_x[i, :] = vec(train_x[:,:,i])
end
sub_y = train_y[1:100]

function main()
    kmeansResult = kMeans(DataFrame(sub_x), 10)
    println(kmeansResult.estimatedClass)
end

main()


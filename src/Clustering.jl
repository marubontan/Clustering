module Clustering

using DataFrames, StatsBase

include("kmeans.jl")
export KMeansResults, kMeans

include("kmedoids.jl")
export kMedoidsResults, kMedoids

include("predict.jl")
export predict

include("dist.jl")
include("type.jl")
include("utils.jl")
export calcDist

end

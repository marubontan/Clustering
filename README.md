# Clustering
This is the package for non-hierarchical clustering algorithms.  

## Install

On Julia's package console, by following, you can install the package.

```julia
add git@github.com:marubontan/Clustering.git
```

## Usage

This is the simple example of kMeans clustering.  

```julia
using DataFrames
using Distributions
using KMeans

function makeData()
   eye = zeros(2, 2)
   for i = 1:2
       eye[i, i] = 1.0
   end
   groupOne = rand(MvNormal([10.0, 10.0], 5.0 * eye), 100)
   groupTwo = rand(MvNormal([0.0, 0.0], 10 * eye), 100)
   return vcat(groupOne, groupTwo)
end

function main()
   data = makeData()

   kmeansResult = kMeans(DataFrame(data), 2; initializer="kmeans++")
end

main()
```

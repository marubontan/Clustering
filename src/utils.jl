using DataFrames
include("./dist.jl")

function judgeConvergence(arrayA::Array, arrayB::Array)
    return arrayA == arrayB
end


function makeValuesProbabilistic(values::Array)
    return values / sum(values)
end


function dataFrameToDict(data::DataFrame)

    indexDataDict = Dict{Int, Array{Float64, 1}}()
    for i in 1:nrow(data)
        indexDataDict[i] = vec(convert(Array, data[i, :]))
    end

    return indexDataDict
end


function calcDistBetweenCenterAndDataPoints(data::Dict, centerIndex::Int)
    center = data[centerIndex]
    distanceDict = Dict{Int, Float64}()
    for pair in data
        distanceDict[pair[1]] = calcDist(center, pair[2])
    end
    return distanceDict
end


function makeDictValueProbabilistic(data::Dict)
    vals = [v for v in values(data)]
    valsSum = sum(vals)

    for pair in data
        data[pair[1]] = pair[2] / valsSum
    end

    return data
end


function dataFrame2JaggedArray(data::DataFrame)
    returnArray = Array{Array}(undef, nrow(data))
    for i in 1:nrow(data)
        returnArray[i] = vec(convert(Array, data[i,:]))
    end
    return returnArray
end


"""
    calcDist(sourcePoint, destPoint; method="euclidean")

Calculate the distance between sourcePoint and destPoint by the method.

# Argument
- `sourcePoint::Array`: Array to show the one point.
- `destPoint::Array`: Array to show the  one point.
- `method::String`: distance function for calculation. euclidean,minkowski.

# Examples
```julia-calcDist
julia> calcDist([1,2,3], [2,3,4]; method="minkowski")
3
```
"""
function calcDist(sourcePoint::Array,
                  destPoint::Array;
                  method="euclidean")

    if method == "euclidean"
        return euclidean(sourcePoint, destPoint)
    elseif method == "minkowski"
        return minkowski(sourcePoint, destPoint)
    end
end

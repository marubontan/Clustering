function judgeConvergence(arrayA::Array, arrayB::Array)
    return arrayA == arrayB
end


function makeValuesProbabilistic(values::Array)
    return values / sum(values)
end


function stochasticallyPickUp(values::Array,
                              probs::Array{Float64},
                              n::Int)

    indexProb = Dict{Int, Float64}()
    for key in 1:length(values)
        indexProb[key] = probs[key]
    end

    pickedValues = []
    for _ in 1:n
        border = rand(1)[1]

        sum = 0
        for pair in indexProb
            sum += pair[2]
            if sum > border
                push!(pickedValues, pair[1])

                # TODO: it's bad hack to delte the loop target in the loop
                delete!(indexProb, pair[1])

                denominator = 1 - pair[2]
                for (key,val) in indexProb
                    indexProb[key] = val / denominator
                end
                break
            end
        end
    end
    return pickedValues
end


function dataFrameToDict(data::DataFrame)

    indexDataDict = Dict{Int, Array}()
    for i in 1:nrow(data)
        indexDataDict[i] = vec(Array(data[i, :]))
    end

    return indexDataDict
end


function calcDistBetweenCenterAndDataPoints(data::Dict, centerIndex::Int)
    center = data[centerIndex]
    distanceDict = Dict()
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
    returnArray = Array{Array}(nrow(data))
    for i in 1:nrow(data)
        returnArray[i] = vec(Array(data[i,:]))
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


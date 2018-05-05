function judgeConvergence(arrayA, arrayB)
    return arrayA == arrayB
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
function calcDist(sourcePoint::Array, destPoint::Array; method="euclidean")

    if method == "euclidean"
        return euclidean(sourcePoint, destPoint)
    elseif method == "minkowski"
        return minkowski(sourcePoint, destPoint)
    end
end

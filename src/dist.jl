"""
    euclidean(sourcePoint, destPoint)

Calculate the euclidean distance between sourcePoint and destPoint.

# Arguments
- `sourcePoint::Array`: Array to show the one point.
- `destPoint::Array`: Array to show the one point.

# Examples
```julia_euclidean
julia> euclidean([1,2,3], [2,3,4])
1.7320508075688772
```
"""
function euclidean(sourcePoint::Array, destPoint::Array)
    length(sourcePoint) == length(destPoint) || error("The lengths of source and destination points should be same.")

    sum = 0
    for i in 1:length(sourcePoint)
        sum += (destPoint[i] - sourcePoint[i]) ^ 2
    end
    dist = sqrt(sum)
    return dist
end

"""
    minkowski(sourcePoint, destPoint)

Calculate the minkowski distance between sourcePoint and destPoint.

# Arguments
- `sourcePoint::Array`: Array to show the one point.
- `destPoint::Array`: Array to show the one point.

# Examples
```julia_minkowski
julia> minkowski([1,2,3], [2,3,4])
3
```
"""
function minkowski(sourcePoint::Array, destPoint::Array)
    length(sourcePoint) == length(destPoint) || error("The lengths of source and destination points should be same.")

    sum = 0
    for i in 1:length(sourcePoint)
        sum += abs(destPoint[i] - sourcePoint[i])
    end
    return sum
end

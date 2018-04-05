function euclidean(sourcePoint, destPoint)
    sum = 0
    for i in 1:length(sourcePoint)
        sum += (destPoint[i] - sourcePoint[i]) ^ 2
    end
    dist = sqrt(sum)
    return dist
end

function minkowski(sourcePoint, destPoint)
    sum = 0
    for i in 1:length(sourcePoint)
        sum += abs(destPoint[i] - sourcePoint[i])
    end
    return sum
end

import math

numRoomsPos = [3, 2, 2, 2]
numRoomsNeg = [0, 2, 4, 1]

aptCondPos = [2, 2, 3, 2]
aptCondNeg = [1, 2, 2, 2]

distPos = [2, 5, 2]
distNeg = [5, 1, 1]

pricePos = [7, 1, 1]
priceNeg = [2, 2, 3]

def calculateEntropy(posCount, negCount):
    if posCount == 0 or negCount == 0:
        return 0.0
    totCount = posCount + negCount
    pPlus = posCount / totCount
    pMinus = negCount / totCount
    return -pPlus * math.log2(pPlus) - pMinus * math.log2(pMinus)

def calculateInformationGain(variablePosCounts, variableNegCounts):
    interimEntropy = []
    expectedEntropy = 0.0
    totalPosLabels = 0
    totalNegLabels = 0
    totalLabels = 0

    for i in range(0, len(variablePosCounts)):
        totalPosLabels += variablePosCounts[i]
        totalNegLabels += variableNegCounts[i]
        interimEntropy.append(calculateEntropy(variablePosCounts[i], variableNegCounts[i]))

    totalLabels = totalPosLabels + totalNegLabels

    for i in range(0, len(variablePosCounts)):
        expectedEntropy += ((variablePosCounts[i] + variableNegCounts[i]) / totalLabels) * interimEntropy[i]

    return calculateEntropy(totalPosLabels, totalNegLabels) - expectedEntropy

print(calculateInformationGain(aptCondPos, aptCondNeg))

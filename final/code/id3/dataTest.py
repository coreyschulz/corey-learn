from data import Data
import math
import numpy as np
DATA_DIR = 'data/'

POS_LABEL = 'like'
NEG_LABEL = 'dislike'

allTestData = Data(fpath = DATA_DIR + 'simple_train.csv')
allTestAttributes = list(allTestData.attributes.keys())

def preProcessData(allData):
    dataManager = {}
    allDataAttributes = list(allTestData.attributes.keys())

    for attribute in allDataAttributes:
        dataManager[attribute] = {}
        attributePosVals = allData.attributes[attribute].possible_vals

        for posVal in attributePosVals:
            dataManager[attribute][posVal] = [0, 0]

        for attribLabel in allData.get_column([attribute, 'label']):
            if NEG_LABEL == attribLabel[1]: ## false
                dataManager[attribute][attribLabel[0]][0] += 1
            else:
                dataManager[attribute][attribLabel[0]][1] += 1

    return dataManager

def calculateEntropy(negCount, posCount):
    if posCount == 0 or negCount == 0:
        return 0.0
    totCount = posCount + negCount
    pPlus = posCount / totCount
    pMinus = negCount / totCount
    return -pPlus * math.log2(pPlus) - pMinus * math.log2(pMinus)

def calculateAttributeInformationGain(counts):
    posValues = []
    negValues = []
    interimEntropy = []
    expectedEntropy = 0.0
    totalPosLabels = 0
    totalNegLabels = 0

    for classifier in counts.values():
        neg = classifier[0]
        pos = classifier[1]
        totalNegLabels += neg
        totalPosLabels += pos
        negValues.append(neg)
        posValues.append(pos)

    for i in range(0, len(posValues)):
        interimEntropy.append(calculateEntropy(negValues[i], posValues[i]))

    totalLabels = totalPosLabels + totalNegLabels

    for i in range(0, len(posValues)):
        expectedEntropy += ((posValues[i] + negValues[i]) / totalLabels) * interimEntropy[i]

    return calculateEntropy(totalNegLabels, totalPosLabels) - expectedEntropy

def calculateAllAttributesInformationGain(dataManager):
    attributesInformationGain = {}

    for attribute, val in dataManager.items():
        attributesInformationGain[attribute] = calculateAttributeInformationGain(val)

    return attributesInformationGain







print(allTestAttributes)
print(allTestData.attributes['num-rooms'].index)
print(allTestData.attributes['num-rooms'].possible_vals)
## print(allTestData.get_column(['distance', 'label']))
processedData = preProcessData(allTestData)
print(processedData)

print(calculateAllAttributesInformationGain(processedData))



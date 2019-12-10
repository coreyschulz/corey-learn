from statistics import *
from copy import *
import numpy as np
import math

class Node:
    def __init__(self, attributeName, depth = 0, informationGain = 0):
        self.depth = depth
        self.children = {}
        self.attributeName = attributeName
        self.informationGain = informationGain

    def addChild(self, attribute, child=None):
        self.children[attribute] = child

    def getAttributeName(self):
        return self.attributeName

    def getInformationGain(self):
        return self.informationGain

    def getChildren(self):
        return self.children

    def setDepth(self, depth):
        self.depth = depth

    def getDepth(self):
        return self.depth

class DecisionTree:
    def __init__(self, trainingData, testData):
        self.trainingData = trainingData
        self.testData = testData
        self.decisionTree = Node('undefined')

    def convergeTree(self):
        id3(self.trainingData, self.trainingData.attributes, self.trainingData.get_column('label'))

    def convergeTreeLimit(self, limit):
        self.decisionTree = (self.trainingData, self.trainingData.attributes, self.trainingData.get_column('label'))
        limitTreeDepth(self.decisionTree, limit)

    def testOverallError(self):
        error, depth = getOverallError(self.testData, self.decisionTree)


def id3(dataObj: object, attributes, labels, depth = -1):
    label = labels[0]
    sameLabel = True

    for i in range(len(labels)):
        if labels[i] != label:
            sameLabel = False
            break

    if sameLabel:
        return Node(label, depth + 1)

    attributeName, ig = getBestAttribute(dataObj)

    root = Node(attributeName, 1, ig)
    for v in dataObj.get_attribute_possible_vals(attributeName):
        newDataObj = dataObj.get_row_subset(attributeName, v, dataObj.raw_data)

        if len(newDataObj) == 0:
            try:
                commonValue = mode(dataObj.get_column('label', dataObj.raw_data))
            except StatisticsError:
                commonValue = label
            root.addChild(v, Node(commonValue, depth + 2))
        else:
            newAttribute = deepcopy(attributes)
            newAttribute.pop(attributeName)
            try:
                root.addChild(v, id3(newDataObj, newAttribute, newDataObj.get_column('label'), depth + 1))
            except:
                pass

    return root

def returnMostCommon(dataObject, col):
    try:
        returner = mode(dataObject.get_column(col, dataObject.raw_data))
    except StatisticsError:
        returner = "it's a tie!"
    return returner


def calculateInformationGain(labelCount, attributeGroupedByLabel):
    totalEntropy = calculateSingleEntropy(list(labelCount.values())[0], list(labelCount.values())[1])
    expectedEntropy = attributeExpectedEntropy(labelCount, attributeGroupedByLabel)

    return totalEntropy - expectedEntropy

def calculateSingleEntropy(negCount, posCount):
    if posCount == 0 or negCount == 0:
        return 0.0
    totCount = posCount + negCount
    pPlus = posCount / totCount
    pMinus = negCount / totCount
    return -pPlus * math.log2(pPlus) - pMinus * math.log2(pMinus)

def attributeExpectedEntropy(labelCount, attributeGroupedByLabel):
    attributeEntropy = []

    for attribute in attributeGroupedByLabel.items():
        fraction = sum(attribute[1].values()) / sum(labelCount.values())
        attributeValueEntropy = fraction * calculateSingleEntropy(list(attribute[1].values())[1],
                                                                  list(attribute[1].values())[0])

        attributeEntropy.append(attributeValueEntropy)

    return sum(attributeEntropy)


def calculateTotalEntropy(dataObject):
    counts = {}
    for label in dataObject.get_column('label'):
        if not label in counts:
            counts[label] = 1
        else:
            counts[label] += 1
    return calculateSingleEntropy(list(counts.values())[0], list(counts.values())[1])

def getBestAttribute(dataObj):
    maxGain = ('', 0.0)

    for i in dataObj.attributes.keys():
        attributeLabelCols = dataObj.get_column([i, 'label'])
        attributeGroupedByLabel = groupAttributeByLabel(dataObj, attributeLabelCols)

        currentGain = calculateInformationGain(groupLabel(dataObj), attributeGroupedByLabel)
        if currentGain >= maxGain[1]:
            maxGain = (i, currentGain)

    return maxGain[0], maxGain[1]


def limitDepthHelper(node, maxDepth):
    currentDepth = node.getDepth()

    if currentDepth >= maxDepth:
        labelValues = []
        getLabelValues(node, labelValues)

        try:
            commonValue = mode(labelValues)
        except StatisticsError:
            commonValue = labelValues[0]

        return Node(commonValue, depth=currentDepth)

    for attribute, child in node.getChildren().items():
        node.addChild(attribute, limitDepthHelper(child, maxDepth))

    return node


def limitTreeDepth(id3Tree, maxDepth):
    root = id3Tree
    if len(id3Tree.getChildren()) == 0:
        return id3Tree

    limitedTree = limitDepthHelper(root, maxDepth)

    return limitedTree

def getLabelValues(node, labelValues):
    if len(node.getChildren()) == 0:
        labelValues.append(node.getAttributeName())
        return

    for attribute, child in node.getChildren().items():
        getLabelValues(child, labelValues)


def groupLabel(dataObj):
    possibleLabelValues = np.unique(dataObj.get_column('label'))
    labelCount = dict(zip(possibleLabelValues, [0] * len(possibleLabelValues)))

    for label in dataObj.get_column('label'):
        labelCount[label] += 1

    return labelCount


def groupAttributeByLabel(dataObj, attributeLabelCols):
    attributeGroupedByLabel = {}

    for attribute, label in attributeLabelCols:

        if attribute not in attributeGroupedByLabel.keys():
            possibleLabelVals = np.unique(dataObj.get_column('label'))
            labelData = dict(zip(possibleLabelVals, [0] * len(possibleLabelVals)))

            attributeGroupedByLabel[attribute] = labelData

        attributeGroupedByLabel[attribute][label] += 1

    return attributeGroupedByLabel


def getOverallError(dataObj, mainRoot):
    maxDepth = 0
    wrongPredictions = 0
    allPredictions = []

    for test in dataObj.raw_data:
        root = mainRoot

        while len(root.getChildren()) != 0:
            currentAttribute = root.getAttributeName()
            attributeIndex = dataObj.get_column_index(currentAttribute)
            nextAttributeValue = test[attributeIndex]

            if nextAttributeValue in root.getChildren().keys():
                root = root.getChildren()[nextAttributeValue]
            else:
                try:
                    commonValue = mode(dataObj.get_column('label', dataObj.raw_data))
                except StatisticsError:
                    commonValue = dataObj.get_column('label', dataObj.raw_data)[0]

                root = Node(commonValue, depth=root.getDepth() + 1)
                if maxDepth < root.getDepth():
                    maxDepth = root.getDepth()
                break

            if maxDepth < root.getDepth():
                maxDepth = root.getDepth()

        attributeIndex = dataObj.get_column_index('label')
        allPredictions.append(root.getAttributeName())
        if test[attributeIndex] != root.getAttributeName():
            wrongPredictions += 1

    trainingError = wrongPredictions / len(dataObj.raw_data) * 100

    return trainingError, maxDepth, allPredictions

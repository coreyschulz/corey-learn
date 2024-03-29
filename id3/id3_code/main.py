import numpy as np
import statistics
from data import Data
import decisionTree as dt
DATA_DIR = 'data/'
FOLD_DIR = 'data/CVfolds/'

trainingData = Data(fpath = DATA_DIR + 'train.csv')
testData = Data(fpath = DATA_DIR + 'test.csv')
depths = [1, 2, 3, 4, 5, 10, 15]

def crossValidationSetup():
    filenames = ['fold' + str(x) for x in range(1, 6)]

    crossTrainData = []
    crossTestData = []

    dataList = []
    for i in range(len(filenames)):
        for index, filename in enumerate(filenames):
            if i == index:
                testCsv = np.loadtxt(FOLD_DIR + filename + '.csv', delimiter=',', dtype=str)
                testObj = Data(data=testCsv)
                crossTestData.append(testObj)
                continue

            dataList.append(np.loadtxt(FOLD_DIR + filename + '.csv', delimiter=',', dtype=str))

        data = np.concatenate(dataList)
        data_obj = Data(data=data)
        crossTrainData.append(data_obj)

    return filenames, crossTrainData, crossTestData


def crossValidation():

    filenames, crossTrainData, crossTestData = crossValidationSetup()

    allAvgAccuracy = []
    for max_depth in depths:
        accuracies = []
        print('Testing Depth Limit: ', max_depth)

        for i in range(len(filenames)):
            crossTree = dt.id3(crossTrainData[i], crossTrainData[i].attributes, crossTrainData[i].get_column('label'))
            crossTreeLimit = dt.limitTreeDepth(crossTree, max_depth)

            error, depth = dt.getOverallError(crossTestData[i], crossTreeLimit)
            accuracies.append(100.0-error)

        avgAccuracy = statistics.mean(accuracies)
        allAvgAccuracy.append(avgAccuracy)
        print("Average accuracy: ", avgAccuracy)
        print("Standard deviation: ", statistics.stdev(accuracies), '\n')

    return allAvgAccuracy

def getMaxAccuracyDepth(accuracies):
    index = accuracies.index(max(accuracies))
    return depths[index]


def main():
    print('Most common label in training data: ', dt.returnMostCommon(testData, 'label'), '\n')

    print('Entropy of the training data: ', dt.calculateTotalEntropy(testData), '\n')

    decisionTree = dt.id3(trainingData, trainingData.attributes, testData.get_column('label'))
    print('Best feature: ', decisionTree.getAttributeName())
    print('    ...and its information gain: ', decisionTree.getInformationGain(), '\n')

    trainingError, trainingDepth = dt.getOverallError(trainingData, decisionTree)
    print('Accuracy on the training set: ', 100 - trainingError, '\n')

    testError, testDepth = dt.getOverallError(testData, decisionTree)
    print('Accuracy on the test set: ', 100 - testError, '\n')

    print('////////// CROSS VALIDATION //////////')
    accuracies = crossValidation()
    bestDepth = getMaxAccuracyDepth(accuracies)
    print('Best depth: ', bestDepth)

    depthDecisionTree = dt.id3(trainingData, trainingData.attributes, trainingData.get_column('label'))
    depthLimitTree = dt.limitTreeDepth(depthDecisionTree, bestDepth)
    err, depth = dt.getOverallError(testData, depthLimitTree)
    print('Accuracy on test set using the best depth: ', 100 - err, '\n')


main()
from libsvm import read_libsvm_default
import csv
import numpy as np
from data import Data
from random_forest import randomForest
import statistics

np.random.seed(15)

## Setup Data:
trainingInputs, trainingLabels, numFeatures = read_libsvm_default('data/data_madelon/madelon_data_train')
testInputs, testLabels, _ = read_libsvm_default('data/data_madelon/madelon_data_test', numFeatures)
trainingInputsArr = trainingInputs.toarray()
testInputsArr = testInputs.toarray()

## Discretize data:
def discreteizeData(nonDiscreteArr):
    means = np.mean(nonDiscreteArr, axis=0)
    for i in range(len(nonDiscreteArr)):
        for j in range(len(nonDiscreteArr[i])):
            if nonDiscreteArr[i][j] <= means[j]:
                nonDiscreteArr[i][j] = 0
            else:
                nonDiscreteArr[i][j] = 1
    return nonDiscreteArr

discreteizeData(trainingInputsArr)

def crossValidateRandomForest():
    f1Inputs, f1Labels, _ = read_libsvm_default('data/data_madelon/folds/fold1')
    f2Inputs, f2Labels, _ = read_libsvm_default('data/data_madelon/folds/fold2')
    f3Inputs, f3Labels, _ = read_libsvm_default('data/data_madelon/folds/fold3')
    f4Inputs, f4Labels, _ = read_libsvm_default('data/data_madelon/folds/fold4')
    f5Inputs, f5Labels, _ = read_libsvm_default('data/data_madelon/folds/fold5')
    allFoldInputArrays = [f1Inputs.toarray(), f2Inputs.toarray(),
                          f3Inputs.toarray(), f4Inputs.toarray(), f5Inputs.toarray()]
    allFoldLabelArrays = [f1Labels, f2Labels, f3Labels, f4Labels, f5Labels]

    for array in allFoldInputArrays:
        discreteizeData(array)
        print(array)

    forestSizes = [10, 50, 100]

    bestForestSize = None
    bestAccuracy = 0

    counter = 1

    everyAccuracy = []

    for forestSize in forestSizes:
        allAccuracies = []
        for i in range(len(allFoldInputArrays)):
            allTrainData = []
            allTrainLabels = []
            for j in range(len(allFoldInputArrays)):
                if j != i:
                    allTrainData.extend(allFoldInputArrays[j])
                    allTrainLabels.extend(allFoldLabelArrays[j])

            print("Hyperparameters: forest size: " + str(forestSize))

            tempforest = randomForest(numFeatures, forestSize)
            tempforest.train(allTrainData, allTrainLabels)
            evaluation = tempforest.evaluate(allFoldInputArrays[i], allFoldLabelArrays[i])
            accuracy = evaluation
            allAccuracies.append(accuracy)
            everyAccuracy.append(accuracy)

        if statistics.mean(allAccuracies) > bestAccuracy:
            bestAccuracy = statistics.mean(allAccuracies)
            bestForestSize = forestSize

    avgAccuracy = statistics.mean(everyAccuracy)
    print("Best forest size: " + str(bestForestSize))
    print("Best accuracy: " + str(bestAccuracy))
    print("Average accuracy: " + str(avgAccuracy))


crossValidateRandomForest()

hachi = randomForest(numFeatures, 50)
hachi.train(trainingInputsArr, trainingLabels)
print("training set: ")
print(hachi.evaluate(trainingInputsArr, trainingLabels))
print("test set: ")
print(hachi.evaluate(testInputsArr, testLabels))


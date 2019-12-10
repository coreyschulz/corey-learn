from libsvm import read_libsvm_default
import csv
import numpy as np
from data import Data
from random_forest import randomForest
import statistics

np.random.seed(4)

## Setup Data:
trainingInputs, trainingLabels, numFeatures = read_libsvm_default('data/data-splits/data.train')
testInputs, testLabels, _ = read_libsvm_default('data/data-splits/data.test', numFeatures)
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



hachi = randomForest(numFeatures, 50)
hachi.train(trainingInputsArr, trainingLabels)
print("training set: ")
print(hachi.evaluate(trainingInputsArr, trainingLabels))
print("test set: ")
print(hachi.evaluate(testInputsArr, testLabels))


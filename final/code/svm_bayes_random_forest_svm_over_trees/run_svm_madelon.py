
from libsvm import *
import math
from svm import *
from random_forest import *
from naive_bayes import naiveBayes
import statistics
from svm_over_trees import *

## Setup Data:
trainingInputs, trainingLabels, numFeatures = read_libsvm('data/data-splits/data.train')
testInputs, testLabels, _ = read_libsvm('data/data-splits/data.test', numFeatures)
trainingInputsArr = trainingInputs.toarray()
testInputsArr = testInputs.toarray()

evalInputs, evalLabels, _ = read_libsvm('data/data-splits/data.eval.anon')
evalInputsArr = evalInputs.toarray()

## Discretize data:
def discreteizeData(nonDiscreteArr):
    means = np.median(nonDiscreteArr, axis=0)
    for i in range(len(nonDiscreteArr)):
        for j in range(len(nonDiscreteArr[i])):
            if nonDiscreteArr[i][j] <= means[j]:
                nonDiscreteArr[i][j] = 0
            else:
                nonDiscreteArr[i][j] = 1
    return nonDiscreteArr

def logizeData(nonDiscreteArr):
    for i in range(len(nonDiscreteArr)):
        for j in range(len(nonDiscreteArr[i])):
            if nonDiscreteArr[i][j] != 0:
                nonDiscreteArr[i][j] = math.log(nonDiscreteArr[i][j])


discreteizeData(trainingInputsArr)
discreteizeData(testInputsArr)
discreteizeData(evalInputsArr)



for i in range(len(trainingLabels)):
    if trainingLabels[i] == 0:
        trainingLabels[i] = -1

for i in range(len(testLabels)):
    if testLabels[i] == 0:
        testLabels[i] = -1

## SVM test:
testSvm = svm(numFeatures, 1, 1000, 2000)
testSvm.train(trainingInputsArr, trainingLabels)
# print("SVM training evaluation: ")
# print(testSvm.evaluate(trainingInputsArr, trainingLabels))
# print("SVM test evaluation: ")
# print(testSvm.evaluate(testInputsArr, testLabels))
testSvm.evaluate(evalInputsArr, evalLabels)

## Naive Bayes Test:
# testBayes = naiveBayes(numFeatures, .5)
# testBayes.train(trainingInputsArr, trainingLabels)
# # print(testBayes.evaluate(trainingInputsArr, trainingLabels))
# # print(testBayes.evaluate(testInputsArr, testLabels))
# testBayes.evaluate(evalInputsArr, evalLabels)

## Random Forest test:
# testRf = randomForest(numFeatures, 100)
# testRf.train(trainingInputsArr, trainingLabels)
# # print(testRf.evaluate(trainingInputsArr, trainingLabels))
# # print(testRf.evaluate(testInputsArr, testLabels))
# testRf.evaluate(evalInputsArr, evalLabels)

## test svm over trees:
# tester = svmOverTrees(numFeatures, 100, 10, 1)
# tester.train(trainingInputsArr, trainingLabels)
# tester.evaluate(evalInputsArr, evalLabels)











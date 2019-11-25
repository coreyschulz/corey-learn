
from libsvm import read_libsvm
from svm import *
import statistics

## Setup Data:
trainingInputs, trainingLabels, numFeatures = read_libsvm('data/data_madelon/madelon_data_train')
testInputs, testLabels, _ = read_libsvm('data/data_madelon/madelon_data_test', numFeatures)
trainingInputsArr = trainingInputs.toarray()
testInputsArr = testInputs.toarray()


def crossValidateSVM():
    f1Inputs, f1Labels, _ = read_libsvm('data/data_madelon/folds/fold1')
    f2Inputs, f2Labels, _ = read_libsvm('data/data_madelon/folds/fold2')
    f3Inputs, f3Labels, _ = read_libsvm('data/data_madelon/folds/fold3')
    f4Inputs, f4Labels, _ = read_libsvm('data/data_madelon/folds/fold4')
    f5Inputs, f5Labels, _ = read_libsvm('data/data_madelon/folds/fold5')
    allFoldInputArrays = [f1Inputs.toarray(), f2Inputs.toarray(),
                          f3Inputs.toarray(), f4Inputs.toarray(), f5Inputs.toarray()]
    allFoldLabelArrays = [f1Labels, f2Labels, f3Labels, f4Labels, f5Labels]

    initLearningRates = [10**1, 10**0, 10**-1, 10**-2, 10**-3, 10**-4]
    regularizations = [10**1, 10**0, 10**-1, 10**-2, 10**-3, 10**-4]

    bestLearningRate = None
    bestRegularization = None
    bestAccuracy = 0

    counter = 1

    everyAccuracy = []

    for rate in initLearningRates:
        for regularization in regularizations:
            allAccuracies = []
            for i in range(len(allFoldInputArrays)):
                allTrainData = []
                allTrainLabels = []
                for j in range(len(allFoldInputArrays)):
                    if j != i:
                        allTrainData.extend(allFoldInputArrays[j])
                        allTrainLabels.extend(allFoldLabelArrays[j])

                print("Hyperparameters: Learning rate: " + str(rate) + " Regularization: " + str(regularization))

                tempsvm = svm(numFeatures, rate, regularization, 100)
                tempsvm.train(allTrainData, allTrainLabels)
                accuracy = tempsvm.evaluate(allFoldInputArrays[i], allFoldLabelArrays[i])
                allAccuracies.append(accuracy)
                everyAccuracy.append(accuracy)

            if statistics.mean(allAccuracies) > bestAccuracy:
                bestAccuracy = statistics.mean(allAccuracies)
                bestLearningRate = rate
                bestRegularization = regularization

    avgAccuracy = statistics.mean(everyAccuracy)
    print("Best rate: " + str(bestLearningRate))
    print("Best reg: " + str(bestRegularization))
    print("Best accuracy: " + str(bestAccuracy))
    print("Average accuracy: " + str(avgAccuracy))

crossValidateSVM()

## SVM test:
testSvm = svm(numFeatures, 0.001, .0001, 100)
testSvm.train(trainingInputsArr, trainingLabels)
print("SVM training evaluation: ")
print(testSvm.evaluate(trainingInputsArr, trainingLabels))
print("SVM test evaluation: ")
print(testSvm.evaluate(testInputsArr, testLabels))





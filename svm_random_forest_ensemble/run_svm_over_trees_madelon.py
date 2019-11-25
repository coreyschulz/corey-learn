from libsvm import *
from svm_over_trees import *
from svm import *
import statistics

np.random.seed(15)

## Setup Data:
trainingInputs, trainingLabels, numFeatures = read_libsvm_default('data/data_madelon/madelon_data_train')
testInputs, testLabels, _ = read_libsvm_default('data/data_madelon/madelon_data_test', numFeatures)
trainingInputsArr = trainingInputs.toarray()
testInputsArr = testInputs.toarray()


def crossValidateSVMOverTrees():
    f1Inputs, f1Labels, _ = read_libsvm_default('data/data_madelon/folds/fold1')
    f2Inputs, f2Labels, _ = read_libsvm_default('data/data_madelon/folds/fold2')
    f3Inputs, f3Labels, _ = read_libsvm_default('data/data_madelon/folds/fold3')
    f4Inputs, f4Labels, _ = read_libsvm_default('data/data_madelon/folds/fold4')
    f5Inputs, f5Labels, _ = read_libsvm_default('data/data_madelon/folds/fold5')
    allFoldInputArrays = [f1Inputs.toarray(), f2Inputs.toarray(),
                          f3Inputs.toarray(), f4Inputs.toarray(), f5Inputs.toarray()]
    allFoldLabelArrays = [f1Labels, f2Labels, f3Labels, f4Labels, f5Labels]

    initLearningRates = [10**0, 10**-1, 10**-2, 10**-3, 10**-4, 10**-5]
    regularizations = [10**1, 10**0, 10**-1, 10**-2, 10**-3, 10**-4, 10**-5]

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

                tempsvmovertrees = svmOverTrees(numFeatures, 1, rate, regularization)
                tempsvmovertrees.train(allTrainData, allTrainLabels)
                accuracy = tempsvmovertrees.evaluate(allFoldInputArrays[i], allFoldLabelArrays[i])
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

crossValidateSVMOverTrees()

tester = svmOverTrees(numFeatures, 2, .001, .001)
tester.train(trainingInputsArr, trainingLabels)
print("testing: ")
print(tester.evaluate(testInputsArr, testLabels))
print("training: ")
print(tester.evaluate(testInputsArr, testLabels))
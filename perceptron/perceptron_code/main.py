from data.libsvm import read_libsvm
from Perceptron import *
import statistics
import json

####################################
## Setup Data
###################
trainingInputs, trainingLabels, numFeatures = read_libsvm('data/data_train')
testInputs, testLabels, _ = read_libsvm('data/data_test', numFeatures)
trainingInputsArr = trainingInputs.toarray()
testInputsArr = testInputs.toarray()
f1Inputs, f1Labels, _ = read_libsvm('data/CVfolds/fold1')
f2Inputs, f2Labels, _ = read_libsvm('data/CVfolds/fold2')
f3Inputs, f3Labels, _ = read_libsvm('data/CVfolds/fold3')
f4Inputs, f4Labels, _ = read_libsvm('data/CVfolds/fold4')
f5Inputs, f5Labels, _ = read_libsvm('data/CVfolds/fold5')
allFoldInputArrays = [f1Inputs.toarray(), f2Inputs.toarray(),
                      f3Inputs.toarray(), f4Inputs.toarray(), f5Inputs.toarray()]
allFoldLabelArrays = [f1Labels, f2Labels, f3Labels, f4Labels, f5Labels]

####################################
## Cross Validation
###################
def crossValidateOneHyperparameter(foldInputArrays, foldLabelArrays, hyperparameter, perceptronType):
    allAccuracies = []
    for i in range(len(foldInputArrays)):
        allTrainData = []
        allTrainLabels = []
        for j in range(len(foldInputArrays)):
            if j != i:
                allTrainData.extend(foldInputArrays[j])
                allTrainLabels.extend(foldLabelArrays[j])

        if perceptronType == "simple":
            tempPerceptron = SimplePerceptron(numFeatures, hyperparameter)
        elif perceptronType == "decaying":
            tempPerceptron = DecayingPerceptron(numFeatures, hyperparameter)
        elif perceptronType == "averaged":
            tempPerceptron = AveragedPerceptron(numFeatures, hyperparameter)
        elif perceptronType == "pocket":
            tempPerceptron = PocketPerceptron(numFeatures, hyperparameter)
        else:
            tempPerceptron = SimplePerceptron(numFeatures, hyperparameter)

        tempPerceptron.train(allTrainData, allTrainLabels)
        tempPerceptron.evaluate(foldInputArrays[i], foldLabelArrays[i])
        allAccuracies.append(tempPerceptron.evaluate(foldInputArrays[i], foldLabelArrays[i]))
    return statistics.mean(allAccuracies)

def crossValidate(foldInputArrays, foldLabelArrays, perceptronType):
    params = [1, 0.1, 0.01]
    allAccuracies = []
    for rate in params:
        tempAccuracy = 1 - crossValidateOneHyperparameter(foldInputArrays, foldLabelArrays, rate, perceptronType)
        allAccuracies.append(tempAccuracy)
    bestIndex = allAccuracies.index(max(allAccuracies))
    print("CROSS VALIDATION - ", perceptronType, "perceptron: ", "##########")
    print("Best hyperparameter: ", params[bestIndex])
    print("Best hyperparameter accuracy: ", allAccuracies[bestIndex])
    return params[bestIndex]



def majorityBaseline(labelSet):
    numOnes = 0
    numNeg = 0
    for label in labelSet:
        if label == 1:
            numOnes += 1
        else:
            numNeg += 1
    if numNeg >= numOnes:
        majorityLabel = -1
        majority = numNeg
    else:
        majorityLabel = 1
        majority = numOnes
    return majorityLabel, (majority / len(labelSet))


def reportBestWorstWords():
    with open('data/vocab_idx.json') as file:
        jsonObj = json.load(file)

    avgPercep = AveragedPerceptron(numFeatures, learningRate=0.01, epochs=100)
    avgPercep.train(trainingInputsArr, trainingLabels)
    weights = np.array(avgPercep.weights)
    topTen = weights.argsort()[-10:][::-1]
    bottomTen = weights.argsort()[:10]

    print('Top ten words with highest weights: (classified as space article)')
    for goodIndex in topTen:
        print(jsonObj[str(goodIndex)])

    print('\nTop ten words with lowest weights: (classified as medical article)')
    for lowIndex in bottomTen:
        print(jsonObj[str(lowIndex)])
    print('\n\n')


def evaluatePerceptronType(pType):
    print("##############", pType.swapcase(), "PERCEPTRON EVALUATION ##################")
    bestParam = crossValidate(allFoldInputArrays, allFoldLabelArrays, pType)

    if pType == "simple":
        tempPerceptron = SimplePerceptron(numFeatures, bestParam, 20)
        verbosePerceptron = SimplePerceptron(numFeatures, bestParam, 20, testInputsArr, testLabels)
    elif pType == "decaying":
        tempPerceptron = DecayingPerceptron(numFeatures, bestParam, 20)
        verbosePerceptron = DecayingPerceptron(numFeatures, bestParam, 20, testInputsArr, testLabels)
    elif pType == "averaged":
        tempPerceptron = AveragedPerceptron(numFeatures, bestParam, 20)
        verbosePerceptron = AveragedPerceptron(numFeatures, bestParam, 20, testInputsArr, testLabels)
    elif pType == "pocket":
        tempPerceptron = PocketPerceptron(numFeatures, bestParam, 20)
        verbosePerceptron = PocketPerceptron(numFeatures, bestParam, 20, testInputsArr, testLabels)

    else:
        print("Error! Invalid perceptron type!")
        return

    tempPerceptron.train(trainingInputsArr, trainingLabels)
    print("\nNumber of updates made on training data (20 epochs): ", tempPerceptron.numUpdates)
    print('Accuracy on training data: ', 1 - tempPerceptron.evaluate(trainingInputsArr, trainingLabels))
    print('Accuracy on test data: ', 1 - tempPerceptron.evaluate(testInputsArr, testLabels))

    print('\nLearning curve: ')
    verbosePerceptron.train(trainingInputsArr, trainingLabels)

    print('\n\n')

def main():
    tempMajority, tempAccuracy = majorityBaseline(trainingLabels)
    print("Majority Baseline Training Data Accuracy\nMajority Label:", tempMajority,
          "\nAccuracy: ", tempAccuracy, '\n')

    tempMajority, tempAccuracy = majorityBaseline(testLabels)
    print("Majority Baseline Test Data Accuracy\nMajority Label:", tempMajority,
          "\nAccuracy: ", tempAccuracy, '\n')

    ## Averaged Perceptron, learningRate=0.01, top 10 best and worst words:
    reportBestWorstWords()

    evaluatePerceptronType("simple")
    evaluatePerceptronType("decaying")
    evaluatePerceptronType("averaged")
    evaluatePerceptronType("pocket")


main()


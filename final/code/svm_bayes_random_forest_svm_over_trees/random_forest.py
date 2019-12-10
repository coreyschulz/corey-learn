import decisionTree as id3
import numpy as np
import csv
from data import Data
from scipy import stats

def get_data_obj(filename):
    data = np.loadtxt(filename, delimiter=',', dtype=str)
    return Data(data=data)

class randomForest:
    def __init__(self, numFeatures, forestSize):
        self.numFeatures = numFeatures
        self.forestSize = forestSize
        self.trees = []
        self.allRandomData = []
        self.overallVotes = []
        self.votesPerExampleInOrder = []

    def generateRandomData(self, inputs, labels):
        for i in range(self.forestSize):
            returnInputs = []
            returnLabels = []
            examplesSet = np.random.randint(0, len(inputs), 100)
            featuresSet = np.random.randint(0, self.numFeatures, 50)

            for exampleIndex in examplesSet:
                currentInputs = inputs[exampleIndex].tolist()
                appender = []
                for featureIndex in featuresSet:
                    appender.append(int(currentInputs[featureIndex]))
                returnInputs.append(appender)
                returnLabels.append(labels[exampleIndex])

            self.allRandomData.append((returnInputs, returnLabels, featuresSet))

    def translateToCsv(self, filepath, inputsArr, labelsArr, includeFeatures):
        with open(filepath + '.csv', mode='w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writeFeaturesArr = ['label']

            for x in includeFeatures:
                writeFeaturesArr.append(str(x))

            writer.writerow(writeFeaturesArr)

            for i in range(len(labelsArr)):
                temprow = [labelsArr[i]]
                for j in range(len(inputsArr[i])):
                    temprow.append(int(inputsArr[i][j]))
                writer.writerow(temprow)

    def train(self, trainingData, trainingLabels):

        self.generateRandomData(trainingData, trainingLabels)

        for i in range(self.forestSize):
            print("Training: " + str(i))
            tempData = self.allRandomData[i][0]
            tempLabels = self.allRandomData[i][1]
            tempFeatures = self.allRandomData[i][2]

            self.translateToCsv("forest_data/train" + str(i), tempData, tempLabels, tempFeatures)

            tempDataObj = get_data_obj("forest_data/train" + str(i) + ".csv")
            tempTree = id3.id3(tempDataObj, tempDataObj.attributes, tempDataObj.get_column('label'))
            id3.limitTreeDepth(tempTree, 1)
            self.trees.append(tempTree)


    def vote(self, dataObject, tempTree):
        _, _, allPredictions = id3.getOverallError(dataObject, tempTree)
        self.overallVotes.append(allPredictions)

    def evaluate(self, testInputs, testLabels):
        overallErrors = 0
        features = []
        for x in range(self.numFeatures):
            features.append(x)
        self.translateToCsv("forest_data/test", testInputs, testLabels, features)

        trainingDataObj = get_data_obj("forest_data/test.csv")

        for i in range(self.forestSize):
            self.vote(trainingDataObj, self.trees[i])

        for i in range(len(testLabels)):
            allVotesForLabel = []
            for treeDecisions in self.overallVotes:
                allVotesForLabel.append(treeDecisions[i])
            self.votesPerExampleInOrder.append(allVotesForLabel)
            npVotes = np.array(allVotesForLabel)
            majority = stats.mode(npVotes)
            prediction = int(majority[0][0])
            print(prediction)
            if prediction != int(testLabels[i]):
                overallErrors += 1


        return 1 - (overallErrors / len(testLabels))
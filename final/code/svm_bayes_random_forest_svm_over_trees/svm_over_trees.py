from random_forest import *
from svm import *

class svmOverTrees:
    def __init__(self, numFeatures, forestSize, learningRate, regularization):
        self.numFeatures = numFeatures
        self.forestSize = forestSize
        self.learningRate = learningRate
        self.regularization = regularization
        self.forestObject = randomForest(numFeatures, forestSize)
        self.svmObject = svm(numFeatures, learningRate, regularization, 1000)

    def train(self, trainData, trainLabels):
        self.forestObject.train(trainData, trainLabels)
        self.forestObject.evaluate(trainData, trainLabels)
        for i in range(len(self.forestObject.votesPerExampleInOrder)):
            self.forestObject.votesPerExampleInOrder[i] = [int(i) for i in self.forestObject.votesPerExampleInOrder[i]]
        trainArray = np.array(self.forestObject.votesPerExampleInOrder)
        self.svmObject.train(trainData, trainLabels)

    def evaluate(self, testData, testLabels):
        for i in range(len(self.forestObject.votesPerExampleInOrder)):
            self.forestObject.votesPerExampleInOrder[i] = [int(i) for i in self.forestObject.votesPerExampleInOrder[i]]
        testArray = np.array(self.forestObject.votesPerExampleInOrder)

        return self.svmObject.evaluate(testData, testLabels)



import numpy as np
import random
import copy

class SimplePerceptron:

    def __init__(self, numInputs, learningRate=0.01, epochs=10, testArr = None, testLabels = None):
        random.seed(1)
        self.epochs = epochs
        self.learningRate = learningRate
        self.weights = [random.uniform(-0.01, 0.01) for _ in range(numInputs)]
        self.bias = random.uniform(-0.01, 0.01)
        self.numUpdates = 0
        self.testArr = testArr
        self.testLabels = testLabels


    def predict(self, inputs):
        return np.sign(np.dot(inputs, self.weights) + self.bias)

    def train(self, trainingInputs, inputLabels):
        features = copy.deepcopy(trainingInputs)
        labels = copy.deepcopy(inputLabels)
        comb = list(zip(features, labels))

        for e in range(self.epochs):
            random.shuffle(comb)

            for inputs, label in comb:
                yPrime = self.predict(inputs)
                if label != yPrime:
                    self.numUpdates += 1
                    self.weights += self.learningRate * label * inputs
                    self.bias += self.learningRate * label

            if self.testArr is not None:
                print("Epoch", e+1, "; Accuracy: ", 1-self.evaluate(self.testArr, self.testLabels))


        return self.weights

    def evaluate(self, testInputs, testLabels):
        totErrors = 0
        for inputs, label in zip(testInputs, testLabels):
            if label != self.predict(inputs):
                totErrors += 1
        return totErrors / len(testLabels)


class DecayingPerceptron:

    def __init__(self, numInputs, learningRate=0.01, epochs=10, testArr=None, testLabels=None):
        random.seed(1)
        self.epochs = epochs
        self.learningRate = learningRate
        self.learningRateMod = 0
        self.weights = [random.uniform(-0.01, 0.01) for _ in range(numInputs)]
        self.bias = random.uniform(-0.01, 0.01)
        self.numUpdates = 0
        self.testArr = testArr
        self.testLabels = testLabels

    def predict(self, inputs):
        return np.sign(np.dot(inputs, self.weights) + self.bias)

    def train(self, trainingInputs, trainingLabels):
        features = copy.deepcopy(trainingInputs)
        labels = copy.deepcopy(trainingLabels)
        comb = list(zip(features, labels))

        for e in range(self.epochs):
            random.shuffle(comb)

            for inputs, label in comb:
                tempLearningRate = self.learningRate / (1 + self.learningRateMod)
                yPrime = self.predict(inputs)
                if label != yPrime:
                    self.numUpdates += 1
                    self.weights += tempLearningRate * label * inputs
                    self.bias += tempLearningRate * label
            self.learningRateMod += 1
            if self.testArr is not None:
                print("Epoch", e+1, "; Accuracy: ", 1-self.evaluate(self.testArr, self.testLabels))

        return self.weights

    def evaluate(self, testInputs, testLabels):
        totErrors = 0
        for inputs, label in zip(testInputs, testLabels):
            if label != self.predict(inputs):
                totErrors += 1
        return totErrors / len(testLabels)

class AveragedPerceptron:

    def __init__(self, numInputs, learningRate=0.01, epochs=10, testArr=None, testLabels=None):
        random.seed(1)
        self.learningRate = learningRate
        self.epochs = epochs
        self.weights = [random.uniform(-0.01, 0.01) for _ in range(numInputs)]
        self.averagedWeights = np.zeros(numInputs)
        self.bias = random.uniform(-0.01, 0.01)
        self.averagedBias = 0
        self.numUpdates = 0
        self.testArr = testArr
        self.testLabels = testLabels


    def trainPredict(self, inputs):
        return np.sign(np.dot(inputs, self.weights) + self.bias)

    def evaluatePredict(self, inputs):
        return np.sign(np.dot(inputs, self.averagedWeights) + self.averagedBias)

    def train(self, trainingInputs, trainingLabels):
        features = copy.deepcopy(trainingInputs)
        labels = copy.deepcopy(trainingLabels)
        comb = list(zip(features, labels))

        for e in range(self.epochs):
            random.shuffle(comb)

            for inputs, label in comb:
                yPrime = self.trainPredict(inputs)
                if label != yPrime:
                    self.numUpdates += 1
                    self.weights += self.learningRate * label * inputs
                    self.bias += self.learningRate * label

                self.averagedWeights += self.weights
                self.averagedBias += self.bias

            if self.testArr is not None:
                print("Epoch", e+1, "; Accuracy: ", 1-self.evaluate(self.testArr, self.testLabels))

        return self.averagedWeights


    def evaluate(self, testInputs, testLabels):
        totErrors = 0
        for inputs, label in zip(testInputs, testLabels):
            if label != self.evaluatePredict(inputs):
                totErrors += 1
        return totErrors / len(testLabels)

class PocketPerceptron:

    def __init__(self, numInputs, learningRate=0.01, epochs=10, testArr=None, testLabels=None):
        random.seed(1)
        self.learningRate = learningRate
        self.epochs = epochs
        self.weights = [random.uniform(-0.01, 0.01) for _ in range(numInputs)]
        self.bias = random.uniform(-0.01, 0.01)
        self.pocketWeights = copy.deepcopy(self.bias)
        self.pocketBias = 0
        self.currentCounter = 0
        self.pocketCounter = 0
        self.hadFirstError = False
        self.numUpdates = 0
        self.testArr = testArr
        self.testLabels = testLabels

    def predict(self, inputs):
        return np.sign(np.dot(inputs, self.weights) + self.bias)

    def pocketPredict(self, inputs):
        return np.sign(np.dot(inputs, self.pocketWeights) + self.pocketBias)

    def train(self, trainingInputs, trainingLabels):
        features = copy.deepcopy(trainingInputs)
        labels = copy.deepcopy(trainingLabels)
        comb = list(zip(features, labels))

        for e in range(self.epochs):
            random.shuffle(comb)

            for inputs, label in comb:
                yPrime = self.predict(inputs)
                if yPrime != label:
                    self.numUpdates += 1
                    if not self.hadFirstError:
                        self.hadFirstError = True
                        self.pocketCounter = self.currentCounter
                    else:
                        if self.currentCounter > self.pocketCounter:
                            self.pocketCounter = self.currentCounter
                            self.pocketWeights = copy.deepcopy(self.weights)
                            self.pocketBias = self.bias
                    self.currentCounter = 0
                    self.weights += self.learningRate * label * inputs
                    self.bias += self.learningRate * label
                else:
                    self.currentCounter += 1

            if self.testArr is not None:
                print("Epoch", e+1, "; Accuracy: ", 1-self.evaluate(self.testArr, self.testLabels))

        return self.weights

    def evaluate(self, testInputs, testLabels):
        totErrors = 0
        for inputs, label in zip(testInputs, testLabels):
            if label != self.pocketPredict(inputs):
                totErrors += 1
        return totErrors / len(testLabels)



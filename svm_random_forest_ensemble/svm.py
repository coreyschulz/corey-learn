import numpy as np
import random
import copy

class svm:

    def __init__(self, numInputs, learningRate=0.01, regularization = 1, epochs=10):
        random.seed(1)
        self.epochs = epochs
        self.learningRate = learningRate
        self.initLearningRate = learningRate
        self.regularization = regularization
        self.weights = np.array([random.uniform(-0.01, 0.01) for _ in range(numInputs)])


    def predict(self, inputs):
        return np.sign(np.dot(inputs, self.weights))

    def train(self, trainingInputs, inputLabels):
        features = copy.deepcopy(trainingInputs)
        labels = copy.deepcopy(inputLabels)
        comb = list(zip(features, labels))

        for e in range(1, self.epochs + 1):
            random.shuffle(comb)

            for inputs, label in comb:

                if label * np.dot(inputs, self.weights) <= 1:
                    self.weights *= (1 - self.learningRate)
                    self.weights += self.learningRate * self.regularization * label * inputs
                else:
                    self.weights *= 1 - self.learningRate

            self.learningRate = self.initLearningRate / (1 + e)


        return self.weights

    def evaluate(self, testInputs, testLabels):
        totErrors = 0
        for inputs, label in zip(testInputs, testLabels):
            if label != self.predict(inputs):
                totErrors += 1
        return 1 - (totErrors / len(testLabels))
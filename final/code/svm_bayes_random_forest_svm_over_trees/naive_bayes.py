import numpy as np
import random
import copy

class naiveBayes:

    def __init__(self, numFeatures, smoothingTerm = 1.0, si = 2):
        self.numFeatures = numFeatures
        self.smoothingTerm = smoothingTerm
        self.si = si
        self.numPosLabels = 0
        self.numNegLabels = 0
        self.posFeatureGivenPosLabel = {}
        self.posFeatureGivenNegLabel = {}
        self.posFeaturePosLabelProb = {}
        self.posFeatureNegLabelProb = {}

        for i in range(numFeatures):
            self.posFeatureGivenPosLabel[i] = 0
            self.posFeatureGivenNegLabel[i] = 0
            self.posFeaturePosLabelProb[i] = 0.0
            self.posFeatureNegLabelProb[i] = 0.0

    def predict(self, inputs):
        overallGivenPos = []
        overallGivenNeg = []
        for i in range(len(inputs)):
            if inputs[i] == 1:
                overallGivenPos.append(self.posFeaturePosLabelProb[i])
                overallGivenNeg.append(self.posFeatureNegLabelProb[i])
            else:
                overallGivenPos.append(1 - self.posFeaturePosLabelProb[i])
                overallGivenNeg.append(1 - self.posFeatureNegLabelProb[i])

        resultGivenPos = np.prod(overallGivenPos)
        resultGivenNeg = np.prod(overallGivenNeg)

        if resultGivenPos >= resultGivenNeg:
            return 1
        else:
            return -1




    def train(self, trainingInputs, inputLabels):
        for i in range(len(inputLabels)):
            if inputLabels[i] == 1:
                self.numPosLabels += 1
                for j in range(len(trainingInputs[i])):
                    if trainingInputs[i][j] == 1:
                        self.posFeatureGivenPosLabel[j] += 1
            else:
                self.numNegLabels += 1
                for j in range(len(trainingInputs[i])):
                    if trainingInputs[i][j] == 1:
                        self.posFeatureGivenNegLabel[j] += 1

        for i in range(self.numFeatures):
            self.posFeaturePosLabelProb[i] = (self.posFeatureGivenPosLabel[i] + self.smoothingTerm) / (self.numPosLabels + self.si * self.smoothingTerm)
            self.posFeatureNegLabelProb[i] = (self.posFeatureGivenNegLabel[i] + self.smoothingTerm) / (self.numNegLabels + self.si * self.smoothingTerm)

    def evaluate(self, testInputs, testLabels):
        numAccurate = 0
        for i in range(len(testInputs)):
            if self.predict(testInputs[i]) == testLabels[i]:
                numAccurate += 1
            print(self.predict(testInputs[i]))
        return numAccurate / len(testInputs)

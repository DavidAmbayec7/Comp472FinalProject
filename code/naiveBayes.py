# naiveBayes.py:
# COMP 472 Final Project
# David Ambayec 40207617
# November 24, 2025

# Description:
#   Implements the Naive Bayes algorithm manually
#   and compares with the Scikit Gaussian Naive
#   Bayes classifier on the dataset of CIFAR-10.


import numpy as np
import os
from sklearn.naive_bayes import GaussianNB as gNBClassifier
from metrics import metrics


class gNB:

    def __init__(self, numClasses):
        self.numClasses = numClasses
        self.means = {}
        self.stds = {}   

    # Calculate mean, std, for each class and each label:
    def calculateByClass(self, feats, labels):
        labels = labels.astype(int)
        self.means = {}
        self.stds = {} 

        # Calculate each mean and std per class i and per feature q:
        for i in np.unique(labels):
            allSamplesOfClass = feats[labels == i]
            self.means[i] = allSamplesOfClass.mean(axis=0)
            self.stds[i] = allSamplesOfClass.std(axis=0)
    

    # Calculate gaussian distribution:
    def guassianProb(self, data, mean, stdeviation):
        eps = 1e-10
        stdeviation += eps

        # source for gaussian distribution equation: [2]
        exponent = np.exp(-(np.pow(data - mean, 2) / (2 * np.pow(stdeviation + eps, 2))))
        return (1 / (np.sqrt(2 * np.pi) * (stdeviation + eps))) * exponent
    

    # Compute a prediction of a label from data:
    def predict(self, data):
        bestGuess = None
        bestProb = -1.0

        for i in self.means.keys():

            mean = self.means[i]
            std  = self.stds[i]

            guess = self.guassianProb(data, mean, std)
            
            probs = np.prod(guess)

            # Assign best guess:
            if bestGuess is None or probs > bestProb:
                bestProb = probs
                bestGuess = i

        return bestGuess
    

    # Compute all predictions from data:
    def prediction(self, data):
        predictions = []

        for i in range(data.shape[0]):
            predictions.append(self.predict(data[i]))     # Predict label of each data element

        # Return all predictions
        return np.array(predictions, dtype=int)

def main():


    # Load features and convert:
    dataPath = os.path.join("..", "features", "pca50Features.npz")
    data = np.load(dataPath)

    trainFeatures = data["trainFeatures"]
    trainLabels = data["trainLabels"]
    testFeatures  = data["testFeatures"]
    testLabels  = data["testLabels"]

    # start Gaussian Naive Bayes manual algorithm:
    gnbManual = gNB(numClasses=10)
    gnbManual.calculateByClass(trainFeatures, trainLabels)
    pred = gnbManual.prediction(testFeatures)

    # Compute metrics with true and predicted values:
    print("Results for manual Naive Bayes:\n")
    metricsM = metrics("Naive Bayes Manual", testLabels, pred, 10)

    # Use scikit gaussianNB:
    sciGNB = gNBClassifier()
    sciGNB.fit(trainFeatures, trainLabels)
    sciPred = sciGNB.predict(testFeatures)
    # Compute metrics with true and predicted values:
    metricsSK = metrics("Naive Bayes from Scikit GNB Classifier", testLabels, sciPred, 10)


if __name__ == "__main__":
    main()
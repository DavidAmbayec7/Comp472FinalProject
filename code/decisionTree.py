# decisionTree.py:
# COMP 472 Final Project
# David Ambayec 40207617
# November 24, 2025

# Description:
#   Develops a manual decision tree classifier,
#   and compares with the Scikit's decision
#   tree on the dataset of CIFAR-10.


import numpy as np
from sklearn.tree import DecisionTreeClassifier as DTC
from metrics import metrics
import os

maxDepth = 50

class dtcManual:

    def __init__(self, numClasses=10):
        self.numClasses = numClasses
        self.treeRoot = None
        self.minSamplesPerSplit = 5

    # Gini function:
    def gini(self, labels):
       
        if labels.size == 0:
            return 0.0
        
        # LLM Prompt 4: how to implement Gini formula
        numCount = np.bincount(labels, minlength=self.numClasses)
        probs = numCount / labels.size
        return 1.0 - np.sum(probs * probs)

    # Make Leaf Node:
    def leaf(self, y):

        # Calculate value:
        numCount = np.bincount(y, minlength=self.numClasses)
        value = np.argmax(numCount)

        # Return the leaf node:
        return {
            "leaf": True,
            "value": int(value)
        }

    # Build tree:
    def buildTree(self, data, labels, depth):

        bestFeat = None
        bestThresh = None
        bestGini = 1.0
        bestSplit = None
        numSamples, numFeatures = data.shape

        # Stop buliding tree if:
        if (depth >= maxDepth or numSamples < self.minSamplesPerSplit or np.unique(labels).size == 1):

            return self.leaf(labels)

        cGini = self.gini(labels)

        # Interate thru all features:
        for i in range(numFeatures):

            # Get unique values of feature:
            vals = data[:, i]
            thresholds = np.unique(vals)

            for q in thresholds:

                lIndex = []
                rIndex = []
                
                # Split for L/R nodes from threshold value:
                for r, s in enumerate(vals):

                    if (s <= q):
                        # Go left:
                        lIndex.append(r)
                    else:
                        # Go right:
                        rIndex.append(r)

                lIndex = np.array(lIndex, dtype=int)
                rIndex = np.array(rIndex, dtype=int)

                

                # Go to next feature if empty:
                if (lIndex.size == 0 or rIndex.size == 0):
                    continue

                # Assign splits:
                labelLeft = labels[lIndex]
                labelRight = labels[rIndex]
                giniL = self.gini(labelLeft)
                giniR = self.gini(labelRight)

                # LLM Prompt 5: how to calculate weighted Gini split:
                giniSplit = (labelLeft.size * giniL + labelRight.size * giniR) / numSamples

                # If split is better, compute new best:
                if giniSplit < bestGini:
                    bestGini = giniSplit
                    bestFeat = i
                    bestThresh = q
                    bestSplit = (lIndex, rIndex)

        # If no improvement make leaf node:
        if ((bestFeat is None) or (bestSplit is None) or (bestGini >= cGini)):
            return self.leaf(labels)

        # Increase for next depth:
        nextDepth = depth + 1

        # Assign nodes for next depth:
        goLeft, goRight = bestSplit
        leftNode = self.buildTree(data[goLeft], labels[goLeft], nextDepth)
        rightNode = self.buildTree(data[goRight], labels[goRight], nextDepth)

        # Return current node:
        return {
            "leaf": False,
            "l": leftNode,
            "r": rightNode,
            "feat": int(bestFeat),
            "thresh": float(bestThresh)
            
        }



    # Predict label from data:
    def predict(self, data):

        node = self.treeRoot


        # Iterate till leaf node:
        while not node["leaf"]:

            # Get threshold and feature from current node:
            thresh = node["thresh"]
            feature = node["feat"]
            
            
            # Assign node to L/R based on threshold:
            if data[feature] <= thresh:
                node = node["l"]

            else:
                node = node["r"]

        return node["value"]

    # Predict all labels:
    def prediction(self, data):
        predictions = []

        # make prediction for each label:
        for i in range(data.shape[0]):

            predictions.append(self.predict(data[i]))

        return np.array(predictions, dtype=int)


def main():


    # Load features and convert:
    dataPath = os.path.join("..", "features", "pca50Features.npz")
    data = np.load(dataPath)

    trainFeatures = data["trainFeatures"]
    trainLabels = data["trainLabels"]
    testFeatures  = data["testFeatures"]
    testLabels  = data["testLabels"]

    # start Decision Tree manual algorithm:
    dtManual = dtcManual(numClasses=10)
    trainLabels = trainLabels.astype(int)
    dtManual.treeRoot = dtManual.buildTree(trainFeatures, trainLabels, 0)     # Build tree from depth 0 
    pred = dtManual.prediction(testFeatures)

    # Compute metrics with true and predicted values:
    print("Results for manual Decision Tree:\n")
    metricsM = metrics("manual Decision Tree", testLabels, pred, 10)

    # Use scikit gaussianNB:
    sciDT = DTC(criterion="gini", max_depth=maxDepth)
    sciDT.fit(trainFeatures, trainLabels)
    sciPred = sciDT.predict(testFeatures)

    # Compute metrics with true and predicted values:
    metricsSK = metrics("Scikit Decision Tree Classifier", testLabels, sciPred, 10)


if __name__ == "__main__":
    main()
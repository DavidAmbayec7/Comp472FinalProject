# metrics.py:
# COMP 472 Final Project
# David Ambayec 40207617
# November 24, 2025

# Description:
#   Computes metrics for a given model


import numpy as np

# Generate confusion matrix:
def confMatrix(x, y, numClasses=10):

    # Make the confusion matrix:
    cMatrix = np.zeros((numClasses, numClasses))

    for i, q in zip(x, y):
        cMatrix[i, q] = cMatrix[i, q] + 1

    return cMatrix

# Generate metrics function:
def metrics(model, trueVal, predVal, numClasses=10):

    cMatrix = confMatrix(trueVal, predVal, numClasses)    # make confusion matrix

    # Print confusion matrix:
    print(f"Size of confusion matrix: {cMatrix.shape} \n")
    print("Confusion Matrix:\n")
    print(f"{cMatrix}\n")
    
    # Calculate accuracy from confusion matrix:
    numCorrect = np.trace(cMatrix)
    totalCorrect = cMatrix.sum()
    accuracy = numCorrect/totalCorrect
    
    allPrecision = []
    allRecall = []
    allFOne = []


    # Calculate true positive, false positive and negative:
    for i in range(numClasses):
        truePos = cMatrix[i, i]
        falsePos = cMatrix[:, i].sum() - truePos
        falseNeg = cMatrix[i, :].sum() - truePos

        # Calculate other metrics:   
        precision = computePrecision(truePos, falsePos)
        recall = computeRecall(truePos, falseNeg)
        fOne = computeFOne(precision, recall)

        # Append metrics to list:
        allPrecision.append(precision)
        allRecall.append(recall)
        allFOne.append(fOne)

    # Print metrics:
    print(f"\n Evaluation results for model: {model} \n")
    print(f"Accuracy: {accuracy:.2f}\n")
    print("\t Class #:\t Precision:\t Recall:\t F1 Score:\t\n")

    for i in range(numClasses):
        n = i + 1
        print(f"\t {n}\t\t {allPrecision[i]:.2f}\t\t {allRecall[i]:.2f}\t\t {allFOne[i]:.2f}\n")


# Calculate precision:
def computePrecision( tPos, fPos):

    if ((tPos > 0)  or (fPos > 0)):
        precision = tPos/(tPos + fPos)    # source for equation: [1]
    else:
        precision = 0.0
        
    return precision

# Calculate recall:
def computeRecall(tPos, fNeg):

    if ((tPos > 0)  or (fNeg > 0)):
        recall = tPos/(tPos + fNeg)    # source for equation: [1]
    else:
        recall = 0.0
        
    return recall

# Calculate F1:
def computeFOne(p, r):
    if ((p > 0)  or (r > 0)):
        f = (2 * p * r)/(p + r)    # source for equation: [1]
    else:
        f = 0.0
        
    return f
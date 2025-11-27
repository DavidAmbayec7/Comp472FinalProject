# Comp472FinalProject
Final Project for Comp 472
David Ambayec 40207617

Code Structure:

1) root/processData.py:
    Downloads CIFAR-10, extracts 512-dimensional features from ResNet-18, applies PCA and stores features in root/features.

2) root/dataset/:
    Directory for downloaded CIFAR-10 dataset.

3) root/features/:
    Directory for processed features from CIFAR-10 dataset

4) root/code/metrics.py:
    All model codes call upon metrics(), sending their confusion matrix. The function then calculate all evaluation metrics (accuracy, precision, recall, F1-score)

4) root/code/naiveBayes.py:
    Model code that implements Gaussian Naive Bayes manually and compares with scikit-learn GaussianNB using metrics().

4) root/code/decisionTree.py:
    Model code that implements a decision tree manually and compares with scikit-learn DecisionTreeClassifier using metrics().

4) root/code/mlp.py:
    Model code that builds and trains a 3 layer MLP (by default) and calls metrics() for evaluation.


How to run code:

1) Run processData.py to download and process CIFAR-10 data
2) Run code/naiveBayes.py for the Gaussian Naive Bayes implementation and evaluation
3) Run code/decisionTree.py for the Decision Tree implementation and evaluation
4) Run code/mlp.py for the Multi-Layer Perceptron implementation and evaluation

*) To avoid downloading and processing CIFAR-10, skip step #1 and download the dataset and features directories from this Google Drive Link: https://drive.google.com/drive/folders/12kEAEyFbvMhyJwVZ6EpIthQweyJXk5zn?usp=sharing

Then copy/paste into the root directory of the project.
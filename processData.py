# processData.py:
# COMP 472 Final Project
# David Ambayec 40207617
# November 24, 2025

# Description:
#   Loads and processes the CIFAR-10 dataset,
#   using 500 training and 100 test images per
#   class (10 classes). 

import torch
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
from PIL import Image
from sklearn.decomposition import PCA

# Load CIFAR-10 Dataset:
def getDataset():

    # Image transform:
    transf = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])  # LLM Prompt 1: how to normalize CIFAR-10 dataset
    ])

    trainingSet = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transf)
    testSet = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transf)
    
    # show 4 training imgs:
    load = torch.utils.data.DataLoader(trainingSet, batch_size=4, shuffle=True)
    img, labels = next(iter(load))
    plt.imshow(torchvision.utils.make_grid(img).permute(1, 2, 0) / 2 + 0.5)
    plt.title(' '.join(trainingSet.classes[label] for label in labels))
    plt.show()

    return trainingSet, testSet

# Store x images in a subset of data:
def storeImgs(currSet, numImgs):

    targs = currSet.targets
    numStored = [0] * 10    # num stored per class (10 classes)
    subsetIndex = []

    for i, l in enumerate(targs):
        
        if (numStored[l] < numImgs):
            subsetIndex.append(i)    # add img to subset index
            numStored[l] = numStored[l] + 1     # increase # stored

        # if all of numStored is filled, exit:
        if all(c >= numImgs for c in numStored):
            break
        
    # Make the subset:
    subset = torch.utils.data.Subset(currSet, subsetIndex)
    return subset


# Remove the final layerof ResNet-18:
def makeFeatureExtractor(device):
    # LLM Prompt 2: how to remove final layer of ResNet-18 to get 512x1

    wts = models.ResNet18_Weights.DEFAULT
    resnet = models.resnet18(weights=wts)

    # Remove final layer:
    feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
    feature_extractor.to(device)
    feature_extractor.eval()
    return feature_extractor

# Extracts 512x1 feature vectors:
def extractFeatures(extractor, device, load):
    # LLM Prompt 3: how to extract 512x1
    featureList = []
    labelsList = []

    with torch.no_grad():
        for imgs, lbls in load:

            imgs = imgs.to(device)
            # (B, 512, 1, 1):
            outputs = extractor(imgs)  
            # (B, 512):
            outputs = outputs.view(outputs.size(0), -1)  
            featureList.append(outputs.cpu().numpy())
            labelsList.append(lbls.numpy())

        features = np.concatenate(featureList, axis=0)
        labels = np.concatenate(labelsList, axis=0)
        return features, labels

def main():
    
    # Use GPU (cuda or else cpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Load the CIFAR-10 dataset:
    training, test = getDataset()

    # Process the subsets:
    subTraining = storeImgs(training, 500)
    subTest = storeImgs(test, 100)

    # Load the subsets:
    loadTraining = torch.utils.data.DataLoader(subTraining, batch_size=256, shuffle=False, num_workers=4)
    loadTest = torch.utils.data.DataLoader(subTest, batch_size=256, shuffle=False, num_workers=4)

    # Make Feature Extractor to remove last ResNet-18 layer:
    extractor = makeFeatureExtractor(device)

    # Start Extrator for 512x1:
    featuresTrain, labelsTrain = extractFeatures(extractor, device, loadTraining)
    featuresTest, labelsTest = extractFeatures(extractor, device, loadTest)
    
    # Store features:
    print("Extracted train features shape: ", featuresTrain.shape)
    print("Extracted test features shape: ", featuresTest.shape)
    print("Saving 512x1 feature vectors")
    np.savez(
        os.path.join("features", "resnet18Features512.npz"),
        trainFeatures=featuresTrain,
        trainLabels=labelsTrain,
        testFeatures=featuresTest,
        testLabels=labelsTest
    )


    # PCA from 512x1 to 50x1:
    pca = PCA(n_components=50)
    trainPCA = pca.fit_transform(featuresTrain)
    testPCA = pca.fit_transform(featuresTest)
    
    # Store PCA model & features:
    print("Saving 50x1 feature vectors")
    np.savez(
        os.path.join("features", "pca50Features.npz"),
        trainFeatures=trainPCA,
        trainLabels=labelsTrain,
        testFeatures=testPCA,
        testLabels=labelsTest
    )


    # Separate PCA save:
    np.save(os.path.join("features", "pcaComponents.npy"), pca.components_)
    np.save(os.path.join("features", "pcaMean.npy"), pca.mean_)


    print("CIFAR-10 Dataset load complete")


if __name__ == "__main__":
    main()
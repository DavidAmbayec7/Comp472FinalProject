# mpl.py:
# COMP 472 Final Project
# David Ambayec 40207617
# November 24, 2025

# Description:
#   Develops a Multi-Layer Perceptron (MLP) model


import numpy as np
import os
from metrics import metrics
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

numEpochs = 50
 

def main():
   
    # Use GPU (cuda else cpu):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Load dataset:
    dataPath = os.path.join("..", "features", "pca50Features.npz")
    data = np.load(dataPath)

    trainFeatures = data["trainFeatures"]
    trainLabels = data["trainLabels"]
    testFeatures  = data["testFeatures"]
    testLabels  = data["testLabels"]

    trainLabels = trainLabels.astype(int)
    testLabels  = testLabels.astype(int)

    # Make tensors:
    torchTrainFeatures = torch.tensor(trainFeatures, dtype=torch.float32)
    torchTrainLabels = torch.tensor(trainLabels, dtype=torch.long)
    torchTestFeatures  = torch.tensor(testFeatures, dtype=torch.float32)
    torchTestLabels  = torch.tensor(testLabels, dtype=torch.long)

    # Make torch dataset:
    torchTrainDataset = TensorDataset(torchTrainFeatures, torchTrainLabels)
    torchTrainDataset  = TensorDataset(torchTestFeatures, torchTestLabels)

    # Make iterable over dataset:
    trainLoader = DataLoader(torchTrainDataset, batch_size=64, shuffle=True)
    testLoader  = DataLoader(torchTrainDataset, batch_size=128, shuffle=False)

   
    mlpModel = nn.Sequential(
        # 3 Layers:
        nn.Linear(50, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 10)

        # 1 Layer:
        # nn.Linear(50, 512),
        # nn.ReLU()

        # 5 Layers:
        # nn.Linear(50, 512),
        # nn.ReLU(),
        # nn.Linear(512, 512),
        # nn.BatchNorm1d(512),
        # nn.ReLU(),
        # nn.Linear(512, 512),
        # nn.BatchNorm1d(512),
        # nn.ReLU(),
        # nn.Linear(512, 512),
        # nn.BatchNorm1d(512),
        # nn.ReLU(),
        # nn.Linear(512, 10)

    ).to(device)    # use GPU (cuda)

    # Assign loss and optimizer functions:
    lossCE = nn.CrossEntropyLoss()
    optimize = torch.optim.SGD(mlpModel.parameters(), momentum=0.9)

    currEpoch = 0   # Set current epoch
    
    # Loop till all epochs done:
    while currEpoch < numEpochs:
        mlpModel.train()
        
        # MLP Training Loop:
        for x, y in trainLoader:
            x = x.to(device)    # use GPU (cuda)
            y = y.to(device)    # use GPU (cuda)

            # Source for training loop: [4]
            optimize.zero_grad()
            predVal = mlpModel(x)
            loss = lossCE(predVal, y)
            loss.backward()
            optimize.step()

        currEpoch += 1  # Training on current epoch done, increment

    # LLM Prompt 6: How to convert predVal and y to numpy:
    predValFinal = predVal.argmax(dim=1).detach().cpu().numpy() 
    trueValFinal = y.detach().cpu().numpy()

    # Compute metrics with true and predicted values:
    metrics("3 Layer MLP:", trueValFinal, predValFinal, numClasses=10)


if __name__ == "__main__":
    main()

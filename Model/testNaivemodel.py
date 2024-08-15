import torch
from torch_geometric.data import DataLoader
import json
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm
import pandas as pd


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse_loss(y_pred, y_true))

# Import the model classes
from autoencoder import GATEncoder, GATDecoder, GraphAutoencoder

# Define the function to load and test the model
def loadModel(model_path):
    # Assuming in_channels and out_channels are known and consistent
    in_channels = 4
    out_channels = 4

    # Load hyperparameters from the model filename
    filename = model_path.split('/')[-1]
    parts = filename.split('_')
    hidden_channels = int(parts[1][1:])
    lr = float(parts[2][2:])
    vectorSize = int(parts[3][3:])
    wd = float(parts[4][2:].replace('e', 'e-'))

    # Reconstruct the model
    encoder = GATEncoder(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=vectorSize)
    decoder = GATDecoder(in_channels=vectorSize, hidden_channels=hidden_channels, out_channels=out_channels)
    model = GraphAutoencoder(encoder, decoder)

    # Load the state dictionary
    model.load_state_dict(torch.load(model_path))

    return model


def loadTestData(path):
    graphDataset = torch.load(path)
    testData = DataLoader(graphDataset, batch_size=1, shuffle=False)

    return testData

def plotVelDistroAtFrames(predictedVals, filename=""):
    velocities = [((p[2]**2 + p[3]**2)**0.5) for p in predictedVals]

    trimmed = 0.1/2

    # Sort velocities and remove the top and bottom 15%
    velocities_sorted = np.sort(velocities)
    lower_bound = int(trimmed * len(velocities_sorted))
    upper_bound = int((1-trimmed) * len(velocities_sorted))
    trimmed_velocities = velocities_sorted[lower_bound:upper_bound]

    # Plot the histogram of the trimmed velocities
    plt.hist(trimmed_velocities, bins=30, density=True)
    plt.title("Velocity Distribution (" + str(trimmed*200) + " % trimmed)")
    plt.xlabel('Velocity')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    if filename:
        plt.savefig("../plots/" + filename + ".png")
    
    plt.show()

def testAndPlot(model, dataloader, name=""):    
    model.eval()
    velDistroAt = [500]
    criterion = torch.nn.MSELoss()  # Assuming RMSELoss is derived from MSELoss
    frame = 0
    losses = []
    frames = []
    velDistroHolder = []
    velDistroHolderExp = []
    idx = 0
    with torch.no_grad():
        for data in tqdm(dataloader):
            if idx > 25:
                x_recon = model(data.x, data.edge_index)
                loss = criterion(x_recon, data.y).sqrt()  # Applying sqrt to get RMSE
                losses.append(loss.item() / len(data.x) / 4)
                x_recon = x_recon[1:] ## Remove the first
                velDistroHolder.extend(x_recon.tolist())
                velDistroHolderExp.extend(data.y[1:].tolist())
                if (frame + 1) in velDistroAt:
                    plotVelDistroAtFrames(velDistroHolder, name+"Predicted")
                    plotVelDistroAtFrames(velDistroHolderExp, name+"Expected")
                frame += 1
                frames.append(frame)
            if frame > 500:
                break
            idx += 1

    # Convert losses to a pandas Series
    loss_series = pd.Series(losses)

    # Calculate the rolling average with a window of 10 frames
    rolling_avg = loss_series.rolling(window=10).mean()

    # Plot the original losses
    plt.plot(frames, losses, label='Loss vs frames')

    # Plot the rolling average
    plt.plot(frames, rolling_avg, label='Rolling Avg (10 frames)', linestyle='--')

    # Add labels and legend
    plt.xlabel('Frames')
    plt.ylabel('Loss (RMSE) WRT ground truth')
    plt.legend()

    # Show plot
    plt.show()

# Path to the saved model
model_path = "../models/model_h8192_lr0.0001_vec64_wd0_e50.pt"

## Change tlo the dataset you create using the physics simulation
testData_path = "../Dataset/graph_dataset_naive.pt"

# Test dataset 
testDataLoader = loadTestData(testData_path)

# Load and test the model
model = loadModel(model_path)

## Plot the data
testAndPlot(model, testDataLoader, "GAT")
import torch
from torch_geometric.data import DataLoader
import json
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse_loss(y_pred, y_true))

# Import the model classes
from autoencoder import GATEncoder, GATDecoder, GraphAutoencoder
from mlp import MLP
from linearReg import LinearRegressionModel

# Define the function to load and test the model
def loadModel(model_path, intermidiateModelPath):
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

    # Instantiate the model
    ## Test LR
    mlp = LinearRegressionModel(vectorSize, vectorSize).to(device)
    
    # Test MLP
    #mlp = MLP(vectorSize, hidden_channels, vectorSize).to(device)


    # Load the model state dictionary
    mlp.load_state_dict(torch.load(intermidiateModelPath, map_location=device))
    mlp.eval()

    return encoder, decoder, mlp


def loadTestData(path):
    graphDataset = torch.load(path)
    testData = DataLoader(graphDataset, batch_size=1, shuffle=False)

    return testData


def testAndPlot(encoder, decoder, mlp, dataloader, name=""):    
    encoder.eval()
    decoder.eval()
    mlp.eval()


    highLightAt = [5, 25, 125]
    criterion = RMSELoss()
    frame = 0
    losses = []
    frames = []
    with torch.no_grad():
        for data in tqdm(dataloader):
            x_recon = encoder(data.x, data.edge_index)
            x_recon = mlp(x_recon)
            x_recon = decoder(x_recon)

            loss = criterion(x_recon, data.y)
            losses.append(loss/len(data.x)/4)
            frame += 1
            frames.append(frame)

            if frame > 625:
                break

    plt.plot(frames, losses, label='Loss vs frames')

    # for highLight in highLightAt:
    #     plt.plot(highLight, losses[highLight-1], 'ro', markersize=10)

    # Add labels and legend
    plt.xlabel('Frames')
    plt.ylabel('Loss(RMSE) WRT ground truth')
    plt.legend()

    # Show plot
    plt.show()

# Path to the saved model

## Load the encoder and decoder
model_path = "../models/model_h8192_lr0.0001_vec64_wd0_e50.pt"

## Load the intermidiate model (KAN/MLP/LR)
mlp_path = "../models/lr_model_e50.pt" 

## Load the test dataset. Generate from simulation.py
testData_path = "../Dataset/graph_dataset_naive.pt"

# Test dataset 
testDataLoader = loadTestData(testData_path)

# Load and test the model
model = loadModel(model_path, mlp_path)

## Plot the data
testAndPlot(model, testDataLoader)
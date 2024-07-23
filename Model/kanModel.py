import torch
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
import json, os
from itertools import product
from autoencoder import GAEEncoder, GAEDecoder, GraphAutoencoder
from tqdm import tqdm
from kan import *

## Set the 
encoded_size = 512
hidden_channels = 200
in_channels = 4

device = "cpu"
if torch.cuda.is_available():
    torch.device("cuda")
    device = "cuda"
else:
    torch.device("cpu")

## Load the dataset
# Load the data from the .pt file
data = torch.load("../Dataset/kanDataSet.pt")

def split_encoded_dataset(encoded_dataset, train_fraction):
    # Extract the inputs and outputs for splitting
    inputs = [data.x for data in encoded_dataset]
    outputs = [data.y for data in encoded_dataset]

    # Split the data into train and test sets
    train_input, test_input, train_output, test_output = train_test_split(
        inputs, outputs, train_size=train_fraction, random_state=42
    )

    # Create the dictionary with the desired keys
    data_split = {
        'train_input': train_input.to(device),
        'test_input': test_input.to(device),
        'train_output': train_output.to(device),
        'test_output': test_output.to(device)
    }

    return data_split

dataset = split_encoded_dataset(data, 0.8)

## Define the model.
intermediateKANModel = KAN(width=[encoded_size, hidden_channels, encoded_size], grid=20, k=3, seed=0)
intermediateKANModel.train(dataset, opt="LBFGS", steps=50, stop_grid_update_step=30)
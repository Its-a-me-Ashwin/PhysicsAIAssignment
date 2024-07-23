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

## Load the dataset
# Load the data from the .pt file
data = torch.load("../Dataset/kanDataSet.pt")

# Assuming data is a dictionary with keys "input" and "output"
inputs = data['input']
outputs = data['output']

# Define the fraction for train/test split
train_fraction = 0.8  # for example, 80% train, 20% test

# Split the data into train and test sets
train_input, test_input, train_output, test_output = train_test_split(
    inputs, outputs, train_size=train_fraction, random_state=42
)

# Create the dictionary with the desired keys
data_split = {
    'train_input': train_input,
    'test_input': test_input,
    'train_output': train_output,
    'test_output': test_output
}


## Define the model.
intermediateKANModel = KAN(width=[encoded_size, hidden_channels, encoded_size], grid=20, k=3, seed=0)
intermediateKANModel.train(dataset, opt="LBFGS", steps=50, stop_grid_update_step=30)
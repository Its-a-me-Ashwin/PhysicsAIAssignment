import torch
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
import json, os
from itertools import product
from tqdm import tqdm
from kan import KAN

## Set the hyper parameters.
## These were the best that we noticed.
encoded_size = 12800
hidden_channels = 200 #, 400
in_channels = 4

device = "cpu"
## Use the GPU if the library supports it. 
## Some versions crash the GPU for some weird reason.
if torch.cuda.is_available():
    torch.device("cuda")
    device = "cuda"
else:
    torch.device("cpu")

## Load the dataset
# Load the data from the .pt file
data = torch.load("../Dataset/kanDataSet.pt", map_location=torch.device(device))

def saKanResults():
    pass

def split_encoded_dataset(encoded_dataset, train_fraction):
    # Extract the inputs and outputs for splitting
    inputs = torch.stack([data.x for data in encoded_dataset])
    outputs = torch.stack([data.y for data in encoded_dataset])

    # Split the data into train and test sets
    train_input, test_input, train_output, test_output = train_test_split(
        inputs, outputs, train_size=train_fraction, random_state=42
    )

    # Create the dictionary with the desired keys
    data_split = {
        'train_input': train_input.view(train_input.size(0), -1).to(device),
        'test_input': test_input.view(test_input.size(0), -1).to(device),
        'train_output': train_output.view(train_output.size(0), -1).to(device),
        'test_output': test_output.view(test_output.size(0), -1).to(device)
    }

    return data_split

dataset = split_encoded_dataset(data, 0.8)

print(dataset["train_input"].shape)
print(dataset["test_input"].shape)

## Define the model.
intermediateKANModel = KAN(width=[encoded_size, hidden_channels, encoded_size], grid=20, k=3, seed=0)

## For old versions
##intermediateKANModel.train(dataset, opt="LBFGS", steps=50, stop_grid_update_step=30)

## For updated version
result = intermediateKANModel.fit(dataset, opt="LBFGS", steps=50, stop_grid_update_step=30)
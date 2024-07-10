import torch
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
import json

# Load the dataset
graph_dataset = torch.load("../Dataset/graph_dataset.pt")

# Split dataset into train, validation, and test sets
train_val_data, test_data = train_test_split(graph_dataset, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_val_data, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)



import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GAEEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GAEEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return x

class GAEDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GAEDecoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class GraphAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GraphAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        x_recon = self.decoder(z, edge_index)
        return x_recon


import os

def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        x_recon = model(data.x, data.edge_index)
        loss = criterion(x_recon, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            x_recon = model(data.x, data.edge_index)
            loss = criterion(x_recon, data.y)
            total_loss += loss.item()
    return total_loss / len(loader)

def test(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            x_recon = model(data.x, data.edge_index)
            loss = criterion(x_recon, data.y)
            total_loss += loss.item()
    return total_loss / len(loader)


from itertools import product

# Hyperparameter ranges
hidden_channels_range = [512, 1024]
learning_rate_range = [0.001, 0.0001]
weight_decay_range = [0, 1e-5]
epochs_range = [50]

# Create combinations of all hyperparameters
hyperparameters = product(hidden_channels_range, learning_rate_range, weight_decay_range, epochs_range)

# Create a directory to save models and results
os.makedirs("../models", exist_ok=True)
results_file = "../models/results.json"

# Load existing results if the file exists
if os.path.exists(results_file):
    with open(results_file, 'r') as f:
        results = json.load(f)
else:
    results = []

for hidden_channels, lr, wd, epochs in hyperparameters:
    # Define model, optimizer, and loss function
    encoder = GAEEncoder(in_channels=4, hidden_channels=hidden_channels, out_channels=4096)
    decoder = GAEDecoder(in_channels=4096, hidden_channels=hidden_channels, out_channels=4)
    model = GraphAutoencoder(encoder, decoder)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    ## Use MSE loss. We can use the L1 norm too..
    ## Try using this in the hyper parameter list to00
    criterion = nn.MSELoss()

    # Train the model
    best_val_loss = float('inf')
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion)
        val_loss = validate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss}, Validation Loss: {val_loss}")

        # Save the model if it has the best validation loss so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = f"../models/model_h{hidden_channels}_lr{lr}_wd{wd}_e{epochs}.pt"
            torch.save(model.state_dict(), best_model_path)
    
    # Test the model
    test_loss = test(model, test_loader, criterion)
    print(f"Test Loss: {test_loss}")

    # Save results
    result = {
        "hidden_channels": hidden_channels,
        "learning_rate": lr,
        "weight_decay": wd,
        "epochs": epochs,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "model_path": best_model_path
    }
    results.append(result)

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
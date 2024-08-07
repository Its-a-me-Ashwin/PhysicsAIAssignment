import torch
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
import json, os
from itertools import product
from tqdm import tqdm
import torch.nn as nn

# Import the new GATEncoder and GATDecoder classes
from GCNautoencoder import GCNEncoder, GCNDecoder, GraphAutoencoder

# Load the dataset
graph_dataset = torch.load("../Dataset/graph_dataset_naive.pt")

# Split dataset into train, validation, and test sets
train_val_data, test_data = train_test_split(graph_dataset, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_val_data, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Training, validating, and testing functions
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

# Hyperparameter ranges
vectorSizeRange = [64, 128, 256]
hidden_channels_range = [1034, 2048, 4096, 8192]
learning_rate_range = [0.001, 0.0001]
weight_decay_range = [0, 1e-5]
maxEpochs = 50
saveEpochOn = 10

# Create combinations of all hyperparameters
hyperparameters = product(hidden_channels_range, learning_rate_range, weight_decay_range, vectorSizeRange)

# Create a directory to save models and results
os.makedirs("../models", exist_ok=True)
results_file = "../models/resultsNaive.json"

# Load existing results if the file exists
if os.path.exists(results_file):
    with open(results_file, 'r') as f:
        results = json.load(f)
else:
    results = []


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse_loss(y_pred, y_true))

for hidden_channels, lr, wd, vectorSize in hyperparameters:
    # Define model, optimizer, and loss function
    encoder = GATEncoder(in_channels=4, hidden_channels=hidden_channels, out_channels=vectorSize)
    decoder = GATDecoder(in_channels=vectorSize, hidden_channels=hidden_channels, out_channels=4)
    model = GraphAutoencoder(encoder, decoder)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = RMSELoss()

    # Track losses for each epoch
    train_losses = []
    val_losses = []

    # Train the model
    best_val_loss = float('inf')
    for epoch in tqdm(range(maxEpochs)):
        train_loss = train(model, train_loader, optimizer, criterion)
        val_loss = validate(model, val_loader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{maxEpochs}, Train Loss: {train_loss}, Validation Loss: {val_loss}")

        # Save the model if it has the best validation loss so far or every saveEpochOn epochs
        if val_loss < best_val_loss or (epoch + 1) % saveEpochOn == 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            best_model_path = f"../models/model_h{hidden_channels}_lr{lr}_vec{vectorSize}_wd{wd}_e{epoch+1}naive.pt"
            torch.save(model.state_dict(), best_model_path)
    
    # Test the model
    test_loss = test(model, test_loader, criterion)
    print("Test Loss:", test_loss)

    # Save results
    result = {
        "hidden_channels": hidden_channels,
        "learning_rate": lr,
        "weight_decay": wd,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "model_path": best_model_path,
        "train_losses": train_losses,
        "encodeed_size": vectorSize,
        "val_losses": val_losses
    }
    results.append(result)

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import json, os
from itertools import product
from tqdm import tqdm

def createMLPDataset(kanDatasetPath):
    # Load the graph dataset
    kan_dataset = torch.load(kanDatasetPath)
    
    inputs = []
    targets = []
    
    for data in kan_dataset:
        inputs.append(data.x)  # Add the node feature matrix (input)
        targets.append(data.y)  # Add the target feature matrix
        
    # Convert lists to tensors
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)
    
    # Create TensorDataset for use with DataLoader
    mlp_dataset = TensorDataset(inputs, targets)
    
    return mlp_dataset

# Define MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the dataset (assuming it is a tensor dataset)
dataset = createMLPDataset("../Dataset/kanDataSet.pt")

# Split dataset into train, validation, and test sets
train_val_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_val_data, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Training, validating, and testing functions
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        inputs, targets = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            inputs, targets = data
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(loader)

def test(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            inputs, targets = data
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(loader)

if __name__ == "__main__":    
    # Hyperparameter ranges
    hidden_size_range = [64, 128, 256]
    learning_rate_range = [0.001, 0.0001]
    weight_decay_range = [0, 1e-5]
    max_epochs = 50
    save_epoch_on = 10

    # Create combinations of all hyperparameters
    hyperparameters = product(hidden_size_range, learning_rate_range, weight_decay_range)

    # Create a directory to save models and results
    os.makedirs("../models", exist_ok=True)
    results_file = "../models/results.json"

    # Load existing results if the file exists
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
    else:
        results = []

    for hidden_size, lr, wd in hyperparameters:
        # Define model, optimizer, and loss function
        input_size = dataset[0][0].shape[1]
        output_size = dataset[0][1].shape[1]

        model = MLP(input_size=input_size, hidden_size=hidden_size, output_size=output_size)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        criterion = nn.MSELoss()

        # Track losses for each epoch
        train_losses = []
        val_losses = []

        # Train the model
        best_val_loss = float('inf')
        for epoch in tqdm(range(max_epochs)):
            train_loss = train(model, train_loader, optimizer, criterion)
            val_loss = validate(model, val_loader, criterion)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print(f"Epoch {epoch+1}/{max_epochs}, Train Loss: {train_loss}, Validation Loss: {val_loss}")

            # Save the model if it has the best validation loss so far or every save_epoch_on epochs
            if (epoch + 1) % save_epoch_on == 0:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                best_model_path = f"../models/mlp_model_h{hidden_size}_lr{lr}_wd{wd}_e{epoch+1}.pt"
                torch.save(model.state_dict(), best_model_path)
        
        # Test the model
        test_loss = test(model, test_loader, criterion)
        print("Test Loss:", test_loss)

        # Save results
        result = {
            "hidden_size": hidden_size,
            "learning_rate": lr,
            "weight_decay": wd,
            "best_val_loss": best_val_loss,
            "test_loss": test_loss,
            "model_path": best_model_path,
            "train_losses": train_losses,
            "val_losses": val_losses
        }
        results.append(result)

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)

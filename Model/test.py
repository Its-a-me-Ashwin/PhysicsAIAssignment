import torch
from torch_geometric.data import DataLoader
import json

# Import the model classes
from autoencoder import GATEncoder, GATDecoder, GraphAutoencoder

# Define the function to load and test the model
def load_and_test_model(model_path, test_loader):
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

    # Evaluate the model on the test set
    criterion = nn.MSELoss()
    test_loss = test(model, test_loader, criterion)
    print(f"Test Loss: {test_loss}")

# Example usage
# test_data = torch.load("../Dataset/test_data.pt")  # Load your test dataset
# test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Path to the saved model
model_path = "../models/model_h1034_lr0.0001_vec256_wd1e-05_e10.pt"

# Load and test the model
#load_and_test_model(model_path, test_loader)

import torch
from torch_geometric.data import DataLoader, Data
import re
from autoencoder import GATEncoder, GATDecoder, GraphAutoencoder

# Load the graph dataset
graph_dataset = torch.load("../Dataset/graph_dataset.pt")

# Define a function to create an encoded dataset using the trained encoder
def create_encoded_dataset(graph_dataset, encoder, device='cpu'):
    encoder.eval()  # Set the encoder to evaluation mode
    encoder.to(device)  # Move the encoder to the specified device
    new_dataset = []

    with torch.no_grad():
        for i in range(len(graph_dataset) - 1):
            data_t = graph_dataset[i].to(device)
            data_t1 = graph_dataset[i + 1].to(device)

            x_t_encoded = encoder(data_t.x, data_t.edge_index)
            x_t1_encoded = encoder(data_t1.x, data_t1.edge_index)

            new_data = Data(x=x_t_encoded, y=x_t1_encoded, edge_index=data_t.edge_index)
            new_dataset.append(new_data)

    return new_dataset

# Define a function to load the model and create the dataset
def createDataSet(modelName):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    full_state_dict = torch.load(modelName, map_location=device)

    # Extract vector size and hidden channels from the filename
    match_vector_size = re.search(r"_vec(\d+)_", modelName)
    match_hidden_channels = re.search(r"_h(\d+)_", modelName)

    if match_vector_size:
        vector_size = int(match_vector_size.group(1))
    else:
        print("Invalid file name: missing vector size.")
        return

    if match_hidden_channels:
        hidden_channels = int(match_hidden_channels.group(1))
    else:
        print("Invalid file name: missing hidden channels.")
        return
    
    # Extract and load the encoder state dictionary
    encoder_state_dict = {k.replace("encoder.", ""): v for k, v in full_state_dict.items() if k.startswith("encoder.")}
    trained_encoder = GATEncoder(in_channels=4, hidden_channels=hidden_channels, out_channels=vector_size)
    trained_encoder.load_state_dict(encoder_state_dict)

    # Create the encoded dataset
    new_dataset = create_encoded_dataset(graph_dataset, trained_encoder, device=device)

    # Save the new dataset
    torch.save(new_dataset, "../Dataset/kanDataSet.pt")

if __name__ == "__main__":
    modelPath = "../models/model_h1034_lr0.0001_vec256_wd1e-05_e10.pt"
    createDataSet(modelPath)

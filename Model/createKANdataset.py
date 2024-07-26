import torch
from autoencoder import *
from torch_geometric.data import Data

graph_dataset = torch.load("../Dataset/graph_dataset.pt")

def create_encoded_dataset(graph_dataset, encoder, device='cpu'):
    # Ensure the encoder is in evaluation mode
    encoder.eval()
    
    # Move the encoder to the specified device
    encoder.to(device)
    
    # List to store the new dataset
    new_dataset = []
    
    # Iterate over the graph dataset to create the new dataset
    with torch.no_grad():
        for i in range(len(graph_dataset) - 1):
            # Get the t and t+1 data points
            data_t = graph_dataset[i]
            data_t1 = graph_dataset[i + 1]
            
            # Move data to the specified device
            data_t = data_t.to(device)
            data_t1 = data_t1.to(device)
            
            # Encode the t and t+1 data points
            x_t_encoded = encoder(data_t.x, data_t.edge_index)
            x_t1_encoded = encoder(data_t1.x, data_t1.edge_index)
            
            # Create a new data object for the encoded dataset
            new_data = Data(x=x_t_encoded, y=x_t1_encoded, edge_index=data_t.edge_index)
            
            # Append the new data object to the new dataset list
            new_dataset.append(new_data)
    
    return new_dataset


def createDataSet(modelName):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    full_state_dict = torch.load("../models/model_h1034_lr0.0001_wd1e-05_e50.pt")

    # Extract the encoder part of the state dictionary
    encoder_state_dict = {k.replace("encoder.", ""): v for k, v in full_state_dict.items() if k.startswith("encoder.")}
    trained_encoder = GATEncoder(in_channels=4, hidden_channels=8192, out_channels=512)

    # Load the encoder state dictionary into the encoder model
    trained_encoder.load_state_dict(encoder_state_dict)

    new_dataset = create_encoded_dataset(graph_dataset, trained_encoder, device=device)

    torch.save(new_dataset, "../Dataset/kanDataSet.pt")

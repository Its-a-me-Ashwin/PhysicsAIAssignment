import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GAEEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GAEEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.encodedVectorSize = out_channels
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return x

class GAEDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GAEDecoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.in_channels = out_channels
    
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

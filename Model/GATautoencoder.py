import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1):
        super(GATEncoder, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=heads)
        self.encodedVectorSize = out_channels
    
    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        return x

class GATDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1):
        super(GATDecoder, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1)
        self.in_channels = out_channels
    
    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
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

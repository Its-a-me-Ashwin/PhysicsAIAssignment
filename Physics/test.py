import torch
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
import json, os
from itertools import product
from tqdm import tqdm
import torch.nn as nn
from torch_geometric.data import Data


graph_dataset = torch.load("../Dataset/graph_dataset.pt")

def create_graph_pairs(graphData):
    graphPairs = []
    for t in range(len(graphData) - 1):
        x = graphData[t]
        y = graphData[t + 1]
        graphPairs.append(Data(x=x.x, edge_index=x.edge_index, edge_attr=x.edge_attr, y=y.x))
    return graphPairs

graph_dataset_naive = create_graph_pairs(graph_dataset)
torch.save(graph_dataset_naive, "../Dataset/graph_dataset_naive.pt")

from itertools import product

in_channels = 4
hidden_channels = [1034, 2048, 4096, 8192]
out_channels = [64, 128, 256]

hyperparameters = product(hidden_channels, out_channels)


def trainable_params_gcn(in_channels, hidden_channels, out_channels):
    # First GCNConv Layer
    first_layer_params = (in_channels * hidden_channels) + hidden_channels
    # Second GCNConv Layer
    second_layer_params = (hidden_channels * out_channels) + out_channels
    # Total parameters
    total_params = first_layer_params + second_layer_params
    return 2*total_params


def trainable_params_gat(in_channels, hidden_channels, out_channels, heads):
    # First GATConv Layer
    first_layer_weight_params = in_channels * (hidden_channels / heads) * heads
    first_layer_attention_params = 2 * (hidden_channels / heads) * heads
    first_layer_bias_params = hidden_channels
    first_layer_params = first_layer_weight_params + first_layer_attention_params + first_layer_bias_params
    
    # Second GATConv Layer
    second_layer_weight_params = hidden_channels * (out_channels / heads) * heads
    second_layer_attention_params = 2 * (out_channels / heads) * heads
    second_layer_bias_params = out_channels
    second_layer_params = second_layer_weight_params + second_layer_attention_params + second_layer_bias_params
    
    # Total parameters
    total_params = first_layer_params + second_layer_params
    return 2*total_params

for hidden_channel, out_channel in hyperparameters:
    print("GAT params h:", hidden_channel, "o:", hidden_channel, "params:", trainable_params_gat(4, hidden_channel, out_channel, 4))

for hidden_channel, out_channel in hyperparameters:
    print("GCN params h:", hidden_channel, "o:", hidden_channel, "params:", trainable_params_gcn(4, hidden_channel, out_channel))


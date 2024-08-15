import torch
import torch.nn as nn
from mlp import *

# Define the Linear Regression model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)



if __name__ == "__main__":
    # Initialize the Linear Regression model
    input_size = 64  # Same as the input size for the MLP
    output_size = 64  # To match the number of parameters

    model = LinearRegressionModel(input_size=input_size, output_size=output_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    criterion = nn.MSELoss()

    # Track losses for each epoch
    train_losses = []
    val_losses = []

    max_epochs = 50
    save_epoch_on = 50

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
            best_model_path = f"../models/lr_model_e{epoch+1}.pt"
            torch.save(model.state_dict(), best_model_path)

    # Test the model
    test_loss = test(model, test_loader, criterion)
    print("Test Loss:", test_loss)

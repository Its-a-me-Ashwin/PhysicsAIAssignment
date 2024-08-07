import json
import matplotlib.pyplot as plt
import os

def plot_and_save_results(json_file, output_dir='plots'):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    with open(json_file, 'r') as f:
        results = json.load(f)

    # Extract data ignoring the weight decay
    data = {}
    for entry in results:
        key = (entry['hidden_channels'], entry['learning_rate'], entry.get('encodeed_size', "512"), len(entry['train_losses']))
        data[key] = {
            'train_losses': entry['train_losses'],
            'val_losses': entry['val_losses'],
            'test_loss': entry['test_loss']
        }

    # Create plots
    for (hidden_channels, lr, vectorSize, epochs), losses in data.items():
        fig, ax = plt.subplots(figsize=(12, 8))
        
        epochs_range = list(range(1, epochs + 1))
        ax.plot(epochs_range, losses['train_losses'], label='Training Loss', color='blue')
        ax.plot(epochs_range, losses['val_losses'], label='Validation Loss', color='orange', linestyle='--')
        
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_title(f'Loss over Epochs (HC={hidden_channels}, LR={lr}, VS={vectorSize})')
        ax.legend()
        ax.grid(True)

        # Save plot to the output directory
        filename = f'HC{hidden_channels}_LR{lr}_VS{vectorSize}_epochs{epochs}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)
        plt.close(fig)



def plotResults():
    pass

def plotKANData():
    pass

def calculateRMSEOverFrames(dataFile=""):

    X = [data]
    Y = 
    

    plt.plot()
    plt.save()
    pass

def calculateLossPerParticle():
    pass

if __name__ == "__main__":
    plot_and_save_results('./results.json')

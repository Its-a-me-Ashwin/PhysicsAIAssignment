# KAN + GAT


## How to run the experiments

### Collecting the dataset
* pip install -r requiirments.txt
* cd Physics
* python simulation.py
* Follow the on screen instructions to build your dataset. YOu may modify it by adding any type of Noise to the collected data.

## Train the autoencoders
* cd Model
* python main.py
* This will train the autoencoders

## Train the Intermidiate Models
* python *modelName*.py

## Testing the accuracies
* python testIntermidaiteModel.py
* python testNaiveModel.py

Repo for AAAI 2025 conference. 

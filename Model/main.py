import numpy as np
from kan import *
from matplotlib import pyplot as plt

## Read the dataset
rawInput = np.load("../Dataset/input.npy")
rawOutput = np.load("../Dataset/output.npy")

'''
Foramt of the input:
[
    [
        hyper parameters of the simulation
        particle 1 position particle 1 velocity
        particle 2 position particle 2 velocity
        ...
    ]
    ....
]

Format of the output:
[
    [
        force on particle 1 
        force on particle 2
        ...
    ]
    ....
]
'''

## We can skip that data cleaning step as the data is obtained from the simulation itself.
## But we need to calmp the inputs and outputs between a range for KAN to be applied.




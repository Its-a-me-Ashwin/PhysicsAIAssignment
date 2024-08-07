import torch
import pygame
import sys
import math
from torch_geometric.data import Data
from particle import Particle

# Screen dimensions
screenWidth = 1240
screenHeight = 800

# Colors
WHITE = (255, 255, 255)

def graphDataToParticles(graphData):
    particles = []
    
    # Extract node features (posX, posY, velX, velY)
    nodeFeatures = graphData.x
    
    # Iterate over each node to create Particle objects
    for node in nodeFeatures:
        posX, posY, velX, velY = node.tolist()
        particle = Particle(posX, posY)
        particle.velocity = [velX, velY]
        particles.append(particle)
    
    return particles

def replaySimulation(graphDataset):
    pygame.init()
    screen = pygame.display.set_mode((screenWidth, screenHeight))
    clock = pygame.time.Clock()
    running = True
    frameIndex = 0
    totalFrames = len(graphDataset)
    
    while running and frameIndex < totalFrames:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
        
        screen.fill(WHITE)
        
        # Convert the current graph data point to particles
        particles = graphDataToParticles(graphDataset[frameIndex])
        
        # Draw particles
        for particle in particles:
            velColor = Particle.getVelColor(particle.velocity[0]**2 + particle.velocity[1]**2, 1)  # Assuming maxAbsVel = 1 for color calculation
            pygame.draw.circle(screen, velColor, (int(particle.position[0]), int(particle.position[1])), particle.radius)
        
        pygame.display.flip()
        clock.tick(60)  # Set frame rate to 60 FPS
        
        frameIndex += 1
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    # Load the graph dataset
    graphDataset = torch.load("../Dataset/graph_dataset.pt")
    
    # Replay the simulation
    replaySimulation(graphDataset)
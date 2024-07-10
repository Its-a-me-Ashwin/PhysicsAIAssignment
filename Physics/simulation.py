from math import pi, sqrt
import random
import pygame
from scipy.spatial import KDTree
import numpy as np
import sys
import math, time
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from particle import Particle

# Screen dimensions
screen_width = 1240
screen_height = 800

# Parameters for the simulation
gravityY = 0  # 1500
gravityX = 0
timePeriod = 120
dt = 1 / timePeriod
numHyperparameters = 4

# Colors
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Some parameters for the particles
numParticles = 196
sRadius = 40
SVolume = math.pi * (sRadius**4) / 6
BOXSize = [screen_width, screen_height]

# Change these to change the behavior
targetDensity = 0.5
pressureMultiplier = 1000

class Simulation:
    def __init__(self, AI=None):
        self.Particles = []
        self.running_simulation = True
        self.maxAbsVel = 0
        self.render = True
        self.recording = False
        self.dataSet = []
        self.inputToModel = []
        self.outputToModel = []
        self.graphData = []
        self.screen = None
        self.AI = AI
        if AI == None:
            self.AIControl = False
        else:
            self.AIControl = True

    def confineToBox(self, particle, boxSize=BOXSize, eta=-1.0):
        if particle.position[0] > boxSize[0]:
            particle.position[0] = boxSize[0]
            particle.velocity[0] *= eta
        if particle.position[1] > boxSize[1]:
            particle.position[1] = boxSize[1]
            particle.velocity[1] *= eta
        if particle.position[0] < 0:
            particle.position[0] = 0
            particle.velocity[0] *= eta
        if particle.position[1] < 0:
            particle.position[1] = 0
            particle.velocity[1] *= eta

    def density2Pressure(self, density: float) -> float:
        error = density - targetDensity
        return error * pressureMultiplier

    def smoothingFunction(self, dist, sRadius=sRadius) -> float:
        if dist > sRadius:
            return 0
        vTemp = sRadius - dist
        return (vTemp**2) / SVolume

    def smoothingFunctionGrad(self, dist, sRadius=sRadius) -> float:
        if dist > sRadius:
            return 0
        temp = 12 / ((sRadius**4) * math.pi)
        return (dist - sRadius) * temp

    def calculateDensityAtPoint(self, particles, samplePoint, kdtree) -> float:
        density = 0.0
        indices = kdtree.query_ball_point(samplePoint, sRadius)
        for idx in indices:
            particle = particles[idx]
            dist = sqrt((particle.position[0] - samplePoint[0])**2 + (particle.position[1] - samplePoint[1])**2)
            influence = self.smoothingFunction(dist)
            density += particle.mass * influence
        if density == 0.0:
            density = 1/1000000000
        return density

    def calculatePressureAtPoint(self, particles, samplePoint, kdtree) -> list[int]:
        pressureForce = [0, 0]
        indices = kdtree.query_ball_point(samplePoint, sRadius)
        for idx in indices:
            particle = particles[idx]
            dist = sqrt((particle.position[0] - samplePoint[0])**2 + (particle.position[1] - samplePoint[1])**2)
            if dist == 0:
                continue
            direction = [(particle.position[0] - samplePoint[0]) / dist, (particle.position[1] - samplePoint[1]) / dist]
            slope = self.smoothingFunctionGrad(dist)
            density = particle.density
            sharedPressure = (self.density2Pressure(density) + self.density2Pressure(particle.density)) / 2
            pressureForce[0] += sharedPressure * direction[0] * slope * particle.mass / density
            pressureForce[1] += sharedPressure * direction[1] * slope * particle.mass / density
        return pressureForce

    def addVelocityAway(self, particles, samplePoint, strength=-1, radius=20):
        for particle in particles:
            dist = sqrt((particle.position[0] - samplePoint[0])**2 + (particle.position[1] - samplePoint[1])**2)
            dist = math.sqrt(dist)
            if dist > radius: continue

            particle.velocity[0] += (particle.position[0] - samplePoint[0]) * strength
            particle.velocity[1] += (particle.position[1] - samplePoint[1]) * strength

    def setup(self):
        self.Particles = Particle.generateGridParticles(20, numParticles, BOXSize)

    def create_graph_data(self):
        particle_data = []
        for particle in self.Particles:
            particle_data.extend([particle.position[0], particle.position[1], particle.velocity[0], particle.velocity[1]])

        num_particles = len(self.Particles)

        particle_data = torch.tensor(particle_data).view(num_particles, 4)
        positions = particle_data[:, :2].numpy()
        velocities = particle_data[:, 2:]

        distances = np.linalg.norm(positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=-1)
        adjacency_matrix = np.zeros((num_particles, num_particles))

        for i in range(num_particles):
            for j in range(num_particles):
                if distances[i, j] <= sRadius and i != j:
                    adjacency_matrix[i, j] = distances[i, j]

        edge_index, edge_weight = dense_to_sparse(torch.tensor(adjacency_matrix, dtype=torch.float))

        node_features = torch.cat([particle_data[:, :2], particle_data[:, 2:]], dim=1)
        edge_attr = edge_weight.view(-1, 1)

        graph_data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=node_features)
        self.graphData.append(graph_data)

    def update(self):
        if self.running_simulation:
            inputForFrame = [gravityX, gravityY, targetDensity, pressureMultiplier]
            outputForFrame = []
            positions = [particle.position for particle in self.Particles]
            kdtree = KDTree(positions)

            for particle in self.Particles:
                particle.predictedPosition[0] = particle.position[0] + particle.velocity[0]*dt
                particle.predictedPosition[1] = particle.position[1] + particle.velocity[1]*dt
                particle.density = self.calculateDensityAtPoint(self.Particles, particle.position, kdtree)

            for idx, particle in enumerate(self.Particles):
                pressureForce = self.calculatePressureAtPoint(self.Particles, particle.position, kdtree)

                if self.recording and not self.AIControl:
                    inputForFrame.append(particle.position[0]/screen_width)
                    inputForFrame.append(particle.position[1]/screen_height)
                    inputForFrame.append(particle.velocity[0]/screen_width)
                    inputForFrame.append(particle.velocity[1]/screen_height)
                    outputForFrame.append((pressureForce[0] / particle.density) * dt)
                    outputForFrame.append((pressureForce[1] / particle.density) * dt)

                if not self.AIControl:
                    particle.velocity[0] -= (pressureForce[0] / particle.density) * dt
                    particle.velocity[1] -= (pressureForce[1] / particle.density) * dt
                    particle.velocity[1] += gravityY * dt
                    particle.velocity[0] += gravityX * dt
                else:
                    ## Add your code here or call a function here that acomplishes the task.
                    ## Input to the AI model can include the current state of the particles and the prvious states
                    ## The output of the model will be to give the new veloities of the particles after the elapsed time dt.
                    predictedVelocity = self.AI(self.Particles)
                    particle.velocity[0] -= predictedVelocity[idx][0]
                    particle.velocity[1] -= predictedVelocity[idx][1]

            if self.recording and not self.AIControl:
                self.inputToModel.append(inputForFrame)
                self.outputToModel.append(outputForFrame)

            self.maxAbsVel = 0
            for particle in self.Particles:
                particle.position[0] += particle.velocity[0] * dt
                particle.position[1] += particle.velocity[1] * dt
                absVelocity = particle.velocity[0]**2 + particle.velocity[1]**2
                if absVelocity > self.maxAbsVel:
                    self.maxAbsVel = absVelocity
                self.confineToBox(particle)

            if self.maxAbsVel == 0:
               self.maxAbsVel = 1 

            for particle in self.Particles:
                inputForFrame.append(particle.velocity[0]/(self.maxAbsVel**0.5))

            # Create graph data for the current frame
            self.create_graph_data()

        if self.render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return -1
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        self.running_simulation = not self.running_simulation
                    if event.key == pygame.K_r:
                        self.Particles = Particle.generateGridParticles(30, numParticles, BOXSize)

                    if event.key == pygame.K_w:
                        self.dataSet = []
                        self.inputToModel = []
                        self.outputToModel = []
                        self.graphData = []
                        self.recording = True

                    if event.key == pygame.K_a:
                        self.AIControl = not self.AIControl

                    if event.key == pygame.K_s:
                        self.inputToModel = np.array(self.inputToModel)
                        self.outputToModel = np.array(self.outputToModel)
                        torch.save(self.graphData, "../Dataset/graph_dataset.pt")

                        np.save("../Dataset/input.npy", self.inputToModel)
                        np.save("../Dataset/output.npy", self.outputToModel)
                        self.recording = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        d = self.calculateDensityAtPoint(self.Particles, (event.pos[0], event.pos[1]), kdtree)
                        print("Density:", d)
                    elif event.button == 3:
                        self.addVelocityAway(self.Particles, (event.pos[0], event.pos[1]))

            self.screen.fill(WHITE)

            for particle in self.Particles:
                velColor = Particle.getVelColor(particle.velocity[0]**2 + particle.velocity[1]**2, self.maxAbsVel)
                pygame.draw.circle(self.screen, velColor, (int(particle.position[0]), int(particle.position[1])), particle.radius)

            pygame.display.flip()
            pygame.time.Clock().tick(timePeriod)
        return 0

    def run(self):
        print("Here are the controls:")
        print("R: reset the simulation")
        print("P: pause the simulation")
        print("W: start recording data")
        print("S: save the recorded data")
        print("press any key to continue")
        input()

        if self.render:
            pygame.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))

        self.setup()
        while True:
            if self.update() == -1:
                break
        if self.render:
            pygame.quit()
        sys.exit()

if __name__ == "__main__":
    simulation = Simulation()
    simulation.run()

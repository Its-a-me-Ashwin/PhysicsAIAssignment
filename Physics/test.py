from math import pi, sqrt
import random
import pygame
import numpy as np
import sys
import math, time
from particle import Particle

# Screen dimensions
screen_width = 1240
screen_height = 800

gravity = 500
timePeriod = 120
dt = 1 / timePeriod

WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

numParticles = 196
sRadius = 15
SVolume = math.pi * (sRadius**4) / 6
BOXSize = [screen_width, screen_height]

targetDensity = 3
pressureMultiplier = 100

Particles = []

def get_grid_coordinates(position, sRadius):
    return int(position[0] // sRadius), int(position[1] // sRadius)

def update_grid(grid, particles, sRadius):
    grid.clear()
    for particle in particles:
        gx, gy = get_grid_coordinates(particle.position, sRadius)
        if (gx, gy) not in grid:
            grid[(gx, gy)] = []
        grid[(gx, gy)].append(particle)

def confineToBox(particle, boxSize=BOXSize, eta=-1.0):
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

def density2Pressure(density: float) -> float:
    error = density - targetDensity
    return error * pressureMultiplier

def smoothingFunctionForViscosity(dist, sRadius=sRadius):
    pass

def smoothingFunctionForViscosityGrad(dist, sRadius=sRadius):
    pass

def smoothingFunction(dist, sRadius=sRadius) -> float:
    if dist > sRadius:
        return 0
    vTemp = sRadius - dist
    return (vTemp**2) / SVolume

def smoothingFunctionGrad(dist, sRadius=sRadius) -> float:
    if dist > sRadius:
        return 0
    temp = 12 / ((sRadius**4) * math.pi)
    return (dist - sRadius) * temp


def get_particles_in_neighboring_grids(samplePoint, grid, sRadius):
    gx, gy = get_grid_coordinates(samplePoint, sRadius)
    neighbors = []
    for i in range(gx - 1, gx + 2):
        for j in range(gy - 1, gy + 2):
            if (i, j) in grid:
                neighbors.extend(grid[(i, j)])
    return neighbors

def calculateDensityAtPoint(particles, samplePoint, grid, sRadius=sRadius) -> float:
    density = 0.0
    neighbors = get_particles_in_neighboring_grids(samplePoint, grid, sRadius)
    for particle in neighbors:
        dist = sqrt((particle.position[0] - samplePoint[0])**2 + (particle.position[1] - samplePoint[1])**2)
        influence = smoothingFunction(dist)
        density += particle.mass * influence
    if density == 0.0:
        density = 1/1000000000
    return density

def calculatePressureAtPoint(particles, samplePoint, grid, sRadius=sRadius) -> list[int]:
    pressureForce = [0, 0]
    neighbors = get_particles_in_neighboring_grids(samplePoint, grid, sRadius)
    for particle in neighbors:
        dist = sqrt((particle.position[0] - samplePoint[0])**2 + (particle.position[1] - samplePoint[1])**2)
        if dist == 0:
            continue
        direction = [(particle.position[0] - samplePoint[0]) / dist, (particle.position[1] - samplePoint[1]) / dist]
        slope = smoothingFunctionGrad(dist)
        density = particle.density
        sharedPressure = (density2Pressure(density) + density2Pressure(particle.density)) / 2
        pressureForce[0] += sharedPressure * direction[0] * slope * particle.mass / density
        pressureForce[1] += sharedPressure * direction[1] * slope * particle.mass / density
    return pressureForce

def addVelocityAway(particles, samplePoint, strength=-1, radius=20):
    for particle in particles:
        dist = sqrt((particle.position[0] - samplePoint[0])**2 + (particle.position[1] - samplePoint[1])**2)
        dist = math.sqrt(dist)
        if dist > radius: continue

        particle.velocity[0] += (particle.position[0] - samplePoint[0]) * strength
        particle.velocity[1] += (particle.position[1] - samplePoint[1]) * strength
        
def setup():
    global Particles, grid
    Particles = Particle.generateGridParticles(7, numParticles, BOXSize)
    grid = {}
    for particle in Particles:
        gx, gy = get_grid_coordinates(particle.position, sRadius)
        if (gx, gy) not in grid:
            grid[(gx, gy)] = []
        grid[(gx, gy)].append(particle)

running_simulation = True
maxAbsVel = 0
render = True


## INPUT
## Gravity, targetDensity, pressureMultiplier, [Position2D, Vlocity2D]*numParticles

## OUTPUT
## Acc2D
 
inputToModel = []
outputToModel = []

if render:
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))

def update():
    global running_simulation, maxAbsVel, Particles, grid
    if running_simulation:
        inputForFrame = [gravity, targetDensity, pressureMultiplier]
        outputForFrame = []
        
        update_grid(grid, Particles, sRadius)

        s = time.time()
        for particle in Particles:
            particle.predictedPosition[0] = particle.position[0] + particle.velocity[0]*dt
            particle.predictedPosition[1] = particle.position[1] + particle.velocity[1]*dt    
            particle.density = calculateDensityAtPoint(Particles, particle.position, grid)
        
        for particle in Particles:
            pressureForce = calculatePressureAtPoint(Particles, particle.position, grid)
            particle.velocity[0] -= (pressureForce[0] / particle.density) * dt
            particle.velocity[1] -= (pressureForce[1] / particle.density) * dt
            particle.velocity[1] += gravity * dt
            absVelocity = particle.velocity[0]**2 + particle.velocity[1]**2


            print((pressureForce[0] / particle.density) * dt)
            inputForFrame.append(particle.position[0]/screen_width)
            outputForFrame.append((pressureForce[0] / particle.density) * dt)
        
        for particle in Particles:
            if maxAbsVel == 0.0:
                inputForFrame.append(1.0)
                continue
            inputForFrame.append(particle.velocity[0]/(maxAbsVel**0.5))

        inputToModel.append(inputForFrame)
        outputToModel.append(outputForFrame[0])

        maxAbsVel = 0
        for particle in Particles:
            particle.position[0] += particle.velocity[0] * dt
            particle.position[1] += particle.velocity[1] * dt
            absVelocity = particle.velocity[0]**2 + particle.velocity[1]**2
            if absVelocity > maxAbsVel:
                maxAbsVel = absVelocity
            confineToBox(particle)
        
    if render:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return -1
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    running_simulation = not running_simulation
                if event.key == pygame.K_r:
                    Particles = Particle.generateGridParticles(7, numParticles, BOXSize)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    d = calculateDensityAtPoint(Particles, (event.pos[0], event.pos[1]), grid)
                    print("Density:", d)
                elif event.button == 3:
                    addVelocityAway(Particles, (event.pos[0], event.pos[1]))

        screen.fill(WHITE)

        for particle in Particles:
            velColor = Particle.getVelColor(particle.velocity[0]**2 + particle.velocity[1]**2, maxAbsVel)
            pygame.draw.circle(screen, velColor, (int(particle.position[0]), int(particle.position[1])), particle.radius)

        pygame.display.flip()
        pygame.time.Clock().tick(timePeriod)
    return 0

if __name__ == "__main__":
    setup()
    while True:
        if update() == -1:
            break
    if render:
        pygame.quit()

    inputToModel = np.array(inputToModel)
    outputToModel = np.array(outputToModel)

    np.save("./input.npy", inputToModel)
    np.save("./output.npy", outputToModel)
    sys.exit()
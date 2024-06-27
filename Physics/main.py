## Authors 
# Ashwin 
# Anio


## This is a very simple particle simualtion that tries to simulate a fluid like substance.
## The simulation tries to maintain an uniform density through out the entire box. 
## In other words each particle wants to be neither too close nor too far form the other particles. 

from math import pi, sqrt
import random
import pygame
from scipy.spatial import KDTree
import numpy as np
import sys
import math, time
from particle import Particle

# Screen dimensions
screen_width = 1240
screen_height = 800

## Parameters for the simulation. 
## Tune it for different behaviour. 
gravity = 0#1500
timePeriod = 120
dt = 1 / timePeriod


## Colors
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

## Some parameters for the particles.
numParticles = 196
sRadius = 40
SVolume = math.pi * (sRadius**4) / 6
BOXSize = [screen_width, screen_height]

## Change these to change the behaviour.
targetDensity = 0.5
pressureMultiplier = 1000

Particles = []

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

def calculateDensityAtPoint(particles, samplePoint, kdtree) -> float:
    density = 0.0
    indices = kdtree.query_ball_point(samplePoint, sRadius)
    for idx in indices:
        particle = particles[idx]
        dist = sqrt((particle.position[0] - samplePoint[0])**2 + (particle.position[1] - samplePoint[1])**2)
        influence = smoothingFunction(dist)
        density += particle.mass * influence
    if density == 0.0:
        density = 1/1000000000
    return density

def calculatePressureAtPoint(particles, samplePoint, kdtree) -> list[int]:
    pressureForce = [0, 0]
    indices = kdtree.query_ball_point(samplePoint, sRadius)
    for idx in indices:
        particle = particles[idx]
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
    global Particles
    Particles = Particle.generateGridParticles(20, numParticles, BOXSize)

running_simulation = True
maxAbsVel = 0
render = True
recording = False
dataSet = []

## INPUT
## Gravity, targetDensity, pressureMultiplier, [Position2D, Vlocity2D]*numParticles

## OUTPUT
## Acc2D
 
inputToModel = []
outputToModel = []

def update():
    global running_simulation, maxAbsVel, Particles
    global dataSet, inputToModel, outputToModel, recording
    if running_simulation:
        inputForFrame = [gravity, targetDensity, pressureMultiplier]
        outputForFrame = []
        s = time.time()
        positions = [particle.position for particle in Particles]
        kdtree = KDTree(positions)

        s = time.time()
        for particle in Particles:
            particle.predictedPosition[0] = particle.position[0] + particle.velocity[0]*dt
            particle.predictedPosition[1] = particle.position[1] + particle.velocity[1]*dt
            particle.density = calculateDensityAtPoint(Particles, particle.position, kdtree)


        for particle in Particles:
            pressureForce = calculatePressureAtPoint(Particles, particle.position, kdtree)

            ## Create the dataset if recording.
            if recording:
                ## Position is always between 0 and 1.
                inputForFrame.append(particle.position[0]/screen_width)
                inputForFrame.append(particle.position[1]/screen_height)
                
                ## Velocity is not scaled, you can scale it if necessary.
                inputForFrame.append(particle.velocity[0]/screen_width)
                inputForFrame.append(particle.velocity[1]/screen_height)
                
                ## The expected prediction is the change in velocity.
                outputForFrame.append((pressureForce[0] / particle.density) * dt)
                outputForFrame.append((pressureForce[1] / particle.density) * dt)


            particle.velocity[0] -= (pressureForce[0] / particle.density) * dt
            particle.velocity[1] -= (pressureForce[1] / particle.density) * dt
            particle.velocity[1] += gravity * dt
            absVelocity = particle.velocity[0]**2 + particle.velocity[1]**2

            
        if recording:
            inputToModel.append(inputForFrame)
            outputToModel.append(outputForFrame[0])

        s = time.time()
        maxAbsVel = 0
        for particle in Particles:
            particle.position[0] += particle.velocity[0] * dt
            particle.position[1] += particle.velocity[1] * dt
            absVelocity = particle.velocity[0]**2 + particle.velocity[1]**2
            if absVelocity > maxAbsVel:
                maxAbsVel = absVelocity
            confineToBox(particle)
            
        for particle in Particles:
            inputForFrame.append(particle.velocity[0]/(maxAbsVel**0.5))

    if render:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return -1
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    running_simulation = not running_simulation
                if event.key == pygame.K_r:
                    Particles = Particle.generateGridParticles(30, numParticles, BOXSize)

                ## Start recording
                if event.key == pygame.K_w:
                    dataSet = []
                    outputToModel = []
                    inputToModel = []
                    recording = True

                ## Stop recording and save as np file. 
                if event.key == pygame.K_s:
                    inputToModel = np.array(inputToModel)
                    outputToModel = np.array(outputToModel)

                    np.save("./input.npy", inputToModel)
                    np.save("./output.npy", outputToModel)
                    recording = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    d = calculateDensityAtPoint(Particles, (event.pos[0], event.pos[1]), kdtree)
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
    print("Here are the controls:")
    print("R: reset the simulation")
    print("P: pause the simulation")
    print("W: start recording data")
    print("S: save the recorded data")
    print("press any key to continue")
    input()

    if render:
        # Initialize pygame
        pygame.init()
        screen = pygame.display.set_mode((screen_width, screen_height))

    setup()
    while True:
        if update() == -1:
            break
    if render:
        pygame.quit()
    sys.exit()
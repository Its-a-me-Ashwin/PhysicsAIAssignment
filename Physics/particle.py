import random

class Particle:
    def __init__(self, x, y, radius=3, mass=1, color=(255,255,255)) -> None:
        self.color = color
        self.radius = radius
        self.position = [x, y]
        self.predictedPosition = [x, y]
        self.mass = mass
        self.velocity = [0, 0]
        self.accelaration = [0, 0]
        self.properties = dict()
        self.density = 0
        self.gradient = 0   


    def __repr__(self) -> str:
        return "Density: " + str(self.density) + " Acc: " + str(self.accelaration)

    @staticmethod
    def generateRandomParticles(numParticles, boxSize, velFactor=100):
        particles = []
        for _ in range(numParticles):
            x = random.uniform(0, boxSize[0])
            y = random.uniform(0, boxSize[1])
            velocity_x = random.uniform(-1, 1)
            velocity_y = random.uniform(-1, 1)
            particle = Particle(x, y)
            particle.velocity = [velocity_x*velFactor, velocity_y*velFactor]
            particles.append(particle)
        return particles
    
    @staticmethod
    def getVelColor(value, maxAbsVel):
        # Ensure value is in the range [0, maxAbsVel]
        value = max(0, min(value, maxAbsVel))
        # Calculate the ratio of the value to the maximum value
        if maxAbsVel == 0:
            ratio = 1
        else:
            ratio = value / maxAbsVel
        # Interpolate between blue and red
        red = int(ratio * 255)
        blue = int((1 - ratio) * 255)
        # The green component is always 0
        green = 0
        return (red, green, blue)
    
    @staticmethod
    def generateGridParticles(spacing, numParticles, boxSize):
        particles = []
        numParticlesPerSide = int(numParticles ** 0.5)  # Number of particles per side in the grid
        if numParticlesPerSide * numParticlesPerSide < numParticles:
            numParticlesPerSide += 1

        gridWidth = (numParticlesPerSide - 1) * spacing
        gridHeight = (numParticlesPerSide - 1) * spacing

        startX = (boxSize[0] - gridWidth) / 2
        startY = (boxSize[1] - gridHeight) / 2

        count = 0
        for i in range(numParticlesPerSide):
            for j in range(numParticlesPerSide):
                if count >= numParticles:
                    break
                x = startX + j * spacing
                y = startY + i * spacing
                particle = Particle(x, y)
                particles.append(particle)
                count += 1
            if count >= numParticles:
                break

        return particles
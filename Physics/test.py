from simulation import Simulation


def example_AI_model(particles):
    output = []
    for p in particles:
        newVel = (0, 0)
        output.append(newVel)
    return output

s = Simulation(example_AI_model)
s.run()
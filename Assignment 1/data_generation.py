import numpy as np

def generateSamples(mean, sd, size):
    return np.random.normal(loc=mean, scale=sd, size=size)
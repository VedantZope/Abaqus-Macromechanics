import numpy as np
import pandas as pd

def lossFD(expForce, simForce):
    return np.sqrt(np.mean((expForce - simForce)**2))

def stopFD(expForce, simForce, deviationPercent):
    expForceUpper = expForce * (1 + 0.01 * deviationPercent)
    expForceLower = expForce * (1 - 0.01 * deviationPercent)
    return np.all((simForce >= expForceLower) & (simForce <= expForceUpper))
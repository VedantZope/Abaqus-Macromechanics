import numpy as np
import pandas as pd

def lossFD(targetForce, simForce):
    return np.sqrt(np.mean((targetForce - simForce)**2))

def stopFD(targetForce, simForce, deviationPercent):
    targetForceUpper = targetForce * (1 + 0.01 * deviationPercent)
    targetForceLower = targetForce * (1 - 0.01 * deviationPercent)
    return np.all((simForce >= targetForceLower) & (simForce <= targetForceUpper))
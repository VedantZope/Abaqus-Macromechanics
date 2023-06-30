import numpy as np
import pandas as pd

from scipy.integrate import simpson
# import interp1d
from scipy.interpolate import interp1d

def lossFlow(targetStrain, targetStress, simStress):
    return np.sqrt(np.mean((targetStress - simStress)**2))

def dummy_lossFD(targetDisplacement, targetForce, simForce):
    return np.sqrt(np.mean((targetForce - simForce)**2))

def lossFD(targetDisplacement, targetForce, simForce):
    # Implementing numerical integration of the area bounded by 
    # the two curves and two vertical x axis
    # Define the x-range boundary
    x_start = min(targetDisplacement)
    x_end = max(targetDisplacement)

    # Interpolate the simulated force-displacement curve
    sim_FD_func = interp1d(targetDisplacement, simForce, fill_value="extrapolate")
    target_FD_func = interp1d(targetDisplacement, targetForce, fill_value="extrapolate")

    # Evaluate the two curves at various points within the x-range boundary
    x_values = np.linspace(x_start, x_end, num=1000)

    # Create numpy array flag where the sim is higher than the target
    SimHigherThanTarget = np.array(sim_FD_func(x_values) > target_FD_func(x_values))

    # Find all index where the boolean turns opposite
    turningIndices = np.where(SimHigherThanTarget[:-1] != SimHigherThanTarget[1:])

    if len(turningIndices) == 0:
        if SimHigherThanTarget[0] == True:
            # Sim curve is higher than target curve
            y_upper_curve = sim_FD_func(x_values)
            y_lower_curve = target_FD_func(x_values)
        else:
            # Target curve is higher than sim curve
            y_upper_curve = target_FD_func(x_values)
            y_lower_curve = sim_FD_func(x_values)
        # Calculate the area under each curve using the trapezoidal rule
        area_upper = simpson(y_upper_curve, x_values)
        area_lower = simpson(y_lower_curve, x_values)
        bounded_area = area_upper - area_lower
    else:
        turningIndices = np.insert(turningIndices, 0, 0)
        turningIndices = np.insert(turningIndices, len(turningIndices), len(x_values) - 1)

        #print(turningIndices)
        bounded_area = 0
        for i in range(len(turningIndices) - 1):
            previousIndex, currentIndex = tuple(turningIndices[i:i+2])
            if SimHigherThanTarget[currentIndex] == True:
                # Sim curve is higher than target curve
                y_upper_curve = sim_FD_func(x_values[previousIndex:currentIndex + 1])
                y_lower_curve = target_FD_func(x_values[previousIndex:currentIndex + 1])
            else:
                # Target curve is higher than sim curve
                y_upper_curve = target_FD_func(x_values[previousIndex:currentIndex + 1])
                y_lower_curve = sim_FD_func(x_values[previousIndex:currentIndex + 1])
            # Calculate the area under each curve using the trapezoidal rule
            area_upper = simpson(y_upper_curve, x_values[previousIndex:currentIndex + 1])
            area_lower = simpson(y_lower_curve, x_values[previousIndex:currentIndex + 1])
            bounded_area += area_upper - area_lower
        return bounded_area


def stopFD(targetForce, simForce, deviationPercent):
    targetForceUpper = targetForce * (1 + 0.01 * deviationPercent)
    targetForceLower = targetForce * (1 - 0.01 * deviationPercent)
    return np.all((simForce >= targetForceLower) & (simForce <= targetForceUpper))
import numpy as np
import pandas as pd

from scipy.integrate import simpson
# import interp1d
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def lossFlow(targetStrain, targetStress, simStress):
    return np.sqrt(np.mean((targetStress - simStress)**2))

def dummy_lossFD(targetDisplacement, targetForce, simForce):
    return np.sqrt(np.mean((targetForce - simForce)**2))

def lossFD(targetDisplacement, targetForce, simForce,iteration):
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

def loss_plastic(targetDisplacement, targetForce, simForce, step, epsilon=1e-8):
    """
    This function calculates the loss for the plastic region of the force-displacement curve. 
    The loss is a adaptive weighted sum of the residuals loss, slope loss, and important points penalty.
    The weights adapt as the number of iterations increases.
    """

    # Calculate residuals and corresponding loss
    residuals = targetForce - simForce
    residuals_loss = np.sqrt(np.mean(residuals ** 2))

    # Calculate slopes and corresponding loss
    slope_true = np.diff(targetForce) / np.diff(targetDisplacement)
    slope_pred = np.diff(simForce) / np.diff(targetDisplacement)
    slope_loss = np.sqrt(np.mean((slope_true - slope_pred) ** 2))

    # Calculate first element penalty
    first_true = targetForce[0]
    first_pred = simForce[0]
    first_penalty = abs(first_true - first_pred)

    # Calculate peak penalty
    peak_true = np.argmax(targetForce)
    peak_pred = np.argmax(simForce)
    peak_penalty = abs(targetForce[peak_true] - simForce[peak_pred])

    # Calculate last element penalty
    last_true = targetForce[-1]
    last_pred = simForce[-1]
    last_penalty = abs(last_true - last_pred)

    # Calculate maximum loss values for normalization and add epsilon to prevent division by zero
    max_residuals_loss = np.max(np.sqrt(np.mean((targetForce - targetForce.mean()) ** 2))) + epsilon
    max_slope_loss = np.max(np.sqrt(np.mean((np.diff(targetForce) / np.diff(targetDisplacement)) ** 2))) + epsilon
    max_first_penalty = np.max(abs(targetForce[0] - targetForce.mean())) + epsilon
    max_peak_penalty = np.max(abs(targetForce[np.argmax(targetForce)] - targetForce.mean())) + epsilon
    max_last_penalty = np.max(abs(targetForce[-1] - targetForce.mean())) + epsilon

    # Set fixed weights for residuals and slope loss
    w_res = 2
    w_slope = 0.5

    # Calculate adaptive weights for penalties
    # These weights start low and increase with each step
    w_first = step * 0.001
    w_peak = step * 0.001
    w_last = step * 0.001

    # Normalized and weighted loss calculation
    loss = (w_res * residuals_loss / max_residuals_loss +
            w_slope * slope_loss / max_slope_loss +
            w_first * first_penalty / max_first_penalty +
            w_peak * peak_penalty / max_peak_penalty +
            w_last * last_penalty / max_last_penalty)

    return loss


def lossFD(targetDisplacement, targetForce, simForce,iteration):
    return loss_plastic(targetDisplacement, targetForce, simForce, iteration)


def stopFD_SOO(targetForce, simForce, yieldingIndex, deviationPercent):
    targetForceUpper = targetForce * (1 + 0.01 * deviationPercent)
    targetForceLower = targetForce * (1 - 0.01 * deviationPercent)
    return np.all((simForce[yieldingIndex:] >= targetForceLower[yieldingIndex:]) & (simForce[yieldingIndex:] <= targetForceUpper[yieldingIndex:]))

def stopFD_MOO(targetCurves, simCurves, geometries, yieldingIndices, deviationPercent):
    stopAllCurvesCheck = True
    for geometry in geometries:
        yieldingIndex = yieldingIndices[geometry]
        stopAllCurvesCheck = stopAllCurvesCheck & stopFD_SOO(targetCurves[geometry]['force'][yieldingIndex:], simCurves[geometry]['force'][yieldingIndex:], yieldingIndex, deviationPercent)
    return stopAllCurvesCheck
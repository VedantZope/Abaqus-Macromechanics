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


def calculate_yielding_index(targetDisplacement, targetForce, r2_threshold=0.998):
    """
    This function calculates the end of the elastic (linear) region of the force-displacement curve.
    """
    yielding_index = 0

    # Initialize the Linear Regression model
    linReg = LinearRegression()

    for i in range(2, len(targetDisplacement)):
        linReg.fit(targetDisplacement[:i].reshape(-1, 1), targetForce[:i]) 
        simForce = linReg.predict(targetDisplacement[:i].reshape(-1, 1)) 
        r2 = r2_score(targetForce[:i], simForce) 
        if r2 < r2_threshold:  # If R^2 is below threshold, mark the end of linear region
            yielding_index = i - 1
            break
    return yielding_index

def loss_elastic(targetDisplacement, targetForce, simForce, w_res=0.95, w_slope=0.05):
    """
    This function calculates the loss for the elastic region of the force-displacement curve. 
    The loss is a weighted sum of the residuals loss and slope loss.
    """
    # Calculate residuals and corresponding loss
    residuals = targetForce - simForce
    residuals_loss = np.sqrt(np.mean(residuals ** 2))
    
    # Calculate slopes and corresponding loss
    slope_true = np.diff(targetForce) / np.diff(targetDisplacement)
    slope_pred = np.diff(simForce) / np.diff(targetDisplacement)
    slope_loss = np.sqrt(np.mean((slope_true - slope_pred) ** 2))

    # Weighted loss calculation
    loss = w_res * residuals_loss + w_slope * slope_loss

    return loss

def loss_plastic_mono_mono(targetDisplacement, targetForce, simForce, w_res=0.75, w_slope=0.25):
    """
    This function calculates the loss for the monotonic plastic region of the force-displacement curve.
    The loss is a weighted sum of the residuals loss and slope loss.
    """
    # Calculate residuals and corresponding loss
    residuals = targetForce - simForce
    residuals_loss = np.sqrt(np.mean(residuals ** 2))
    
    # Calculate slopes and corresponding loss
    slope_true = np.diff(targetForce) / np.diff(targetDisplacement)
    slope_pred = np.diff(simForce) / np.diff(targetDisplacement)
    slope_loss = np.sqrt(np.mean((slope_true - slope_pred) ** 2))

    # Weighted loss calculation
    loss = w_res * residuals_loss + w_slope * slope_loss

    return loss

def loss_plastic_peak_peak(targetDisplacement, targetForce, simForce, w_res=0.75, w_slope=0.25, w_peak=0.25 , w_last=0.25):
    """
    This function calculates the loss for the non-monotonic plastic region of the force-displacement curve. 
    The loss is a weighted sum of the residuals loss, slope loss, and peak penalty.
    """
    # Calculate residuals and corresponding loss
    residuals = targetForce - simForce
    residuals_loss = np.sqrt(np.mean(residuals ** 2))

    # Calculate slopes and corresponding loss
    slope_true = np.diff(targetForce) / np.diff(targetDisplacement)
    slope_pred = np.diff(simForce) / np.diff(targetDisplacement)
    slope_loss = np.sqrt(np.mean((slope_true - slope_pred) ** 2))

    # Calculate peak penalty
    peak_true = np.argmax(targetForce)
    peak_pred = np.argmax(simForce)
    peak_penalty = abs(targetForce[peak_true] - simForce[peak_pred])

    # Calculate last element penalty
    last_true = targetForce[-1]
    last_pred = simForce[-1]
    last_penalty = abs(last_true - last_pred)

    # Weighted loss calculation
    loss = w_res * residuals_loss + w_slope * slope_loss + w_peak *peak_penalty + w_last * last_penalty

    return loss

def loss_plastic_peak_mono(targetDisplacement, targetForce, simForce, w_res=0.75, w_slope=0.25, w_peak=0.5):
    # Calculate residuals and corresponding loss
    residuals = targetForce - simForce
    residuals_loss = np.sqrt(np.mean(residuals ** 2))

    # Calculate slopes and corresponding loss
    slope_true = np.diff(targetForce) / np.diff(targetDisplacement)
    slope_pred = np.diff(simForce) / np.diff(targetDisplacement)
    slope_loss = np.sqrt(np.mean((slope_true - slope_pred) ** 2))

    # Calculate peak penalty
    peak_true = np.argmax(targetForce)
    peak_pred = np.argmax(simForce)
    peak_penalty = abs(targetForce[peak_true] - simForce[peak_pred])

    # Weighted loss calculation
    loss = w_res * residuals_loss + w_slope * slope_loss + w_peak *peak_penalty

    return loss


def loss_plastic_mono_peak(targetDisplacement, targetForce, simForce, w_res=0.75, w_slope=0.25):
    # Calculate residuals and corresponding loss
    residuals = targetForce - simForce
    residuals_loss = np.sqrt(np.mean(residuals ** 2))
    
    # Calculate slopes and corresponding loss
    slope_true = np.diff(targetForce) / np.diff(targetDisplacement)
    slope_pred = np.diff(simForce) / np.diff(targetDisplacement)
    slope_loss = np.sqrt(np.mean((slope_true - slope_pred) ** 2))

    # Weighted loss calculation
    loss = w_res * residuals_loss + w_slope * slope_loss
    return loss

# Function to check if a sequence is monotonic
def is_monotonic(y):
    return np.all(np.diff(y) >= 0)

def Vedant_lossFD(targetDisplacement, targetForce, simForce):
    # Calculate the end of the elastic region
    yielding_index = calculate_yielding_index(targetDisplacement, targetForce)

    # Split simulation data into elastic and plastic parts
    simForce_elastic = simForce[:yielding_index]
    simForce_plastic = simForce[yielding_index:]

    # Compute losses
    elastic_loss = loss_elastic(targetDisplacement[:yielding_index], targetForce[:yielding_index], simForce_elastic)

    if (is_monotonic(targetForce[yielding_index:]) and is_monotonic(simForce_plastic)):
        plastic_loss = loss_plastic_mono_mono(targetDisplacement[yielding_index:], targetForce[yielding_index:], simForce_plastic)
    elif (is_monotonic(targetForce[yielding_index:]) and not is_monotonic(simForce_plastic)):
        plastic_loss = loss_plastic_mono_peak(targetDisplacement[yielding_index:], targetForce[yielding_index:], simForce_plastic)
    elif (not is_monotonic(targetForce[yielding_index:]) and is_monotonic(simForce_plastic)):
        plastic_loss = loss_plastic_peak_mono(targetDisplacement[yielding_index:], targetForce[yielding_index:], simForce_plastic)
    elif (not is_monotonic(targetForce[yielding_index:]) and not is_monotonic(simForce_plastic)):
        plastic_loss = loss_plastic_peak_peak(targetDisplacement[yielding_index:], targetForce[yielding_index:], simForce_plastic)

    total_loss = elastic_loss + plastic_loss
    return total_loss

def stopFD(targetForce, simForce, deviationPercent):
    targetForceUpper = targetForce * (1 + 0.01 * deviationPercent)
    targetForceLower = targetForce * (1 - 0.01 * deviationPercent)
    return np.all((simForce[5:] >= targetForceLower[5:]) & (simForce[5:] <= targetForceUpper[5:]))
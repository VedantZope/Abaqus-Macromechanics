import numpy as np
import pandas as pd
import time 

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from scipy.integrate import simpson
from scipy.interpolate import interp1d

# def dummy_lossFD(targetDisplacement, targetForce, simForce):
#     return np.sqrt(np.mean((targetForce - simForce)**2))

##################################################################
# Numerical area integration of the difference between           # 
# the target and simulated force-displacement curves             #
# The area is bounded by the two curves and two vertical x axis  #
##################################################################

def lossFD_hardening(targetDisplacement, targetForce, simForce):
    target_yielding_index = calculate_yielding_index(targetDisplacement, targetForce)
    target_plastic_force = targetForce[target_yielding_index:]
    target_plastic_displacement = targetDisplacement[target_yielding_index:]
    sim_plastic_force = simForce[target_yielding_index:]

    # Implementing numerical integration of the area bounded by the two curves and two vertical x axis
    # Define the x-range boundary
    x_start = min(target_plastic_displacement)
    x_end = max(target_plastic_displacement)

    # Interpolate the simulated force-displacement curve
    sim_FD_func = interp1d(target_plastic_displacement, sim_plastic_force, fill_value="extrapolate")
    target_FD_func = interp1d(target_plastic_displacement, target_plastic_force, fill_value="extrapolate")

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

def lossFD_yielding(targetDisplacement, targetForce, simForce):
    target_yielding_index = calculate_yielding_index(targetDisplacement, targetForce)
    sim_yielding_index = calculate_yielding_index(targetDisplacement, simForce)
    target_yielding_force = targetForce[target_yielding_index]
    sim_yielding_force = simForce[sim_yielding_index]
    #print(target_yielding_index)
    #print(sim_yielding_index)
    #print(target_yielding_force)
    #print(sim_yielding_force)
    #time.sleep(5)
    return np.abs(target_yielding_force - sim_yielding_force)

###########################
# The stopping conditions #
###########################

def stopFD_yielding(targetDisplacement, targetForce, simForce, deviationPercent):
    target_yielding_index = calculate_yielding_index(targetDisplacement, targetForce)
    sim_yielding_index = calculate_yielding_index(targetDisplacement, simForce)
    target_yielding_force = targetForce[target_yielding_index]
    sim_yielding_force = simForce[sim_yielding_index]
    target_yielding_force_upper = target_yielding_force * (1 + 0.01 * deviationPercent)
    target_yielding_force_lower = target_yielding_force * (1 - 0.01 * deviationPercent)
    return np.all((sim_yielding_force >= target_yielding_force_lower) & (sim_yielding_force <= target_yielding_force_upper))

def stopFD_hardening(targetDisplacement, targetForce, simForce, deviationPercent):
    target_yielding_index = calculate_yielding_index(targetDisplacement, targetForce)
    target_plastic_force = targetForce[target_yielding_index:]
    sim_plastic_force = simForce[target_yielding_index:]
    target_plastic_force_upper = target_plastic_force * (1 + 0.01 * deviationPercent)
    target_plastic_force_lower = target_plastic_force * (1 - 0.01 * deviationPercent)
    return np.all((sim_plastic_force >= target_plastic_force_lower) & (sim_plastic_force <= target_plastic_force_upper))

##############################
# Finding the yielding index #
##############################

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
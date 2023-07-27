import sys
import json
import os
import numpy as np
import pandas as pd
import glob
from prettytable import PrettyTable
import copy
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from datetime import datetime
from modules.stoploss import *
import time

import torch
import numpy as np
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils import standardize
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def printLog(message, logPath):
    with open(logPath, 'a+') as logFile:
        logFile.writelines(message + "\n")
    print(message)

def parseBoundsBO(paramInfo):
    paramBounds = {}
    for param in paramInfo:
        paramBounds[param] = (paramInfo[param]['lowerBound'], paramInfo[param]['upperBound'])
    return paramBounds

def is_directory_empty(directory_path):
    return len(os.listdir(directory_path)) == 0

def smoothing_force(force, startIndex, endIndex, iter=20000):
    smooth_force = copy.deepcopy(force)
    for i in range(iter):
        smooth_force = savgol_filter(smooth_force[startIndex:endIndex], 
                                    window_length=5, 
                                    polyorder=3,
                                    mode='interp',
                                    #mode='nearest',
                                    #mode='mirror',
                                    #mode='wrap',
                                    #mode='constant',
                                    #deriv=0,
                                    delta=1
                                    )
        smooth_force = np.concatenate((force[0:startIndex], smooth_force, force[endIndex:]))
    return smooth_force

def interpolatingForce(simDisplacement, simForce, targetDisplacement):
    interpolatingFunction = interp1d(simDisplacement, simForce, fill_value='extrapolate')
    # Interpolate the force
    interpolatedSimForce = interpolatingFunction(targetDisplacement)
    return interpolatedSimForce

def interpolating_FD_Curves(FD_Curves, targetCurve):
    # Interpolate the force from FD_Curves to the target curve
    # FD_Curves is a dictionaries
    # where each element is of form (parameterTuples) => {"displacement": <np.array>, "force": <np.array>}
    # targetCurve is a dictionary of form {"displacement": <np.array>, "force": <np.array>}

    # Create interp1d fitting from scipy
    FD_Curves_copy = copy.deepcopy(FD_Curves)
    for paramsTuple, dispforce in FD_Curves_copy.items():
        simDisp = dispforce["displacement"]
        simForce = dispforce["force"]
        targetDisp = targetCurve["displacement"]
        # Interpolate the force
        FD_Curves_copy[paramsTuple]["force"] = interpolatingForce(simDisp, simForce, targetDisp)
        FD_Curves_copy[paramsTuple]["displacement"] = targetDisp
    return FD_Curves_copy

def interpolatingStress(simStrain, simStress, targetStrain):
    interpolatingFunction = interp1d(simStrain, simStress, fill_value='extrapolate')
    # Interpolate the stress
    interpolatedSimStress = interpolatingFunction(targetStrain)
    return interpolatedSimStress

def interpolating_flowCurves(flowCurves, targetCurve):
    flowCurves_copy = copy.deepcopy(flowCurves)
    for paramsTuple, strainstress in flowCurves_copy.items():
        simStrain = strainstress["strain"]
        simStress = strainstress["stress"]
        targetStrain = targetCurve["strain"]
        # Interpolate the force
        flowCurves_copy[paramsTuple]["stress"] = interpolatingStress(simStrain, simStress, targetStrain)
        flowCurves_copy[paramsTuple]["strain"] = targetStrain
    return flowCurves_copy

def SOO_write_BO_json_log(FD_Curves, targetCurve, yieldingIndex, paramConfig,iteration):
    # Write the BO log file
    # Each line of BO logging json file looks like this
    # {"target": <loss value>, "params": {"params1": <value1>, ..., "paramsN": <valueN>}, "datetime": {"datetime": "2023-06-02 18:26:46", "elapsed": 0.0, "delta": 0.0}}
    # FD_Curves is a dictionaries
    # where each element is of form (parameterTuples) => {"displacement": <np.array>, "force": <np.array>}
    # targetCurve is a dictionary of form {"displacement": <np.array>, "force": <np.array>}

    # Construct the json file line by line for each element in FD_Curves
    # Each line is a dictionary
    
    # Delete the json file if it exists
    if os.path.exists(f"optimizers/logs.json"):
        os.remove(f"optimizers/logs.json")

    for paramsTuple, dispforce in FD_Curves.items():
        # Construct the dictionary
        line = {}
        # Note: BO in Bayes-Opt tries to maximize, so you should use the negative of the loss function.
        line["target"] = -lossFD(targetCurve["displacement"][yieldingIndex:], targetCurve["force"][yieldingIndex:], dispforce["force"][yieldingIndex:],iteration)
        line["params"] = dict(paramsTuple)
        for param in paramConfig:
            line["params"][param] = line["params"][param]/paramConfig[param]["exponent"] 
        line["datetime"] = {}
        line["datetime"]["datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line["datetime"]["elapsed"] = 0.0
        line["datetime"]["delta"] = 0.0

        # json file has not exist yet
        # Write the dictionary to json file
        with open(f"optimizers/logs.json", "a") as file:
            json.dump(line, file)
            file.write("\n")

def MOO_write_BO_json_log(combined_interpolated_params_to_geoms_FD_Curves_smooth, targetCurves, geometries, geometryWeights, yieldingIndices, paramConfig,iteration):
    
    # Delete the json file if it exists
    if os.path.exists(f"optimizers/logs.json"):
        os.remove(f"optimizers/logs.json")

    for paramsTuple, geometriesToForceDisplacement in combined_interpolated_params_to_geoms_FD_Curves_smooth.items():
        # Construct the dictionary
        line = {}
        # Note: BO in Bayes-Opt tries to maximize, so you should use the negative of the loss function.
        loss = 0
        for geometry in geometries:
            yieldingIndex = yieldingIndices[geometry]
            loss += - geometryWeights[geometry] * lossFD(targetCurves[geometry]["displacement"][yieldingIndex:], targetCurves[geometry]["force"][yieldingIndex:], geometriesToForceDisplacement[geometry]["force"][yieldingIndex:],iteration)
        line["target"] = loss
        line["params"] = dict(paramsTuple)
        for param in paramConfig:
            line["params"][param] = line["params"][param]/paramConfig[param]["exponent"] 
        line["datetime"] = {}
        line["datetime"]["datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line["datetime"]["elapsed"] = 0.0
        line["datetime"]["delta"] = 0.0

        # json file has not exist yet
        # Write the dictionary to json file
        with open(f"optimizers/logs.json", "a") as file:
            json.dump(line, file)
            file.write("\n")

def MOO_suggest_BOTORCH(combined_interpolated_params_to_geoms_FD_Curves_smooth, targetCurves, geometries, yieldingIndices, paramConfig,iteration):
    # Calculate losses and prepare data for model
    params = []
    losses = []
    for param_tuple, geom_to_simCurves in combined_interpolated_params_to_geoms_FD_Curves_smooth.items():
        #print(param_tuple)
        params.append([value for param, value in param_tuple])
        # The minus sign is because BOTORCH tries to maximize objectives, but we want to minimize the loss
        loss_iter = []
        for geometry in geometries:
            yieldingIndex = yieldingIndices[geometry]
            loss_iter.append(- lossFD(
                targetCurves[geometry]["displacement"][yieldingIndex:], 
                targetCurves[geometry]["force"][yieldingIndex:], 
                geom_to_simCurves[geometry]["force"][yieldingIndex:],
                iteration
            ))
        losses.append(loss_iter)

    # Convert your data to the tensor(float 64)
    X = torch.tensor(params, dtype=torch.float64)
    Y = torch.stack([torch.tensor(loss, dtype=torch.float64) for loss in losses])

    # Define the bounds of the search space
    lower_bounds = torch.tensor([paramConfig[param]['lowerBound'] * paramConfig[param]['exponent'] for param in paramConfig.keys()]).float()
    upper_bounds = torch.tensor([paramConfig[param]['upperBound'] * paramConfig[param]['exponent'] for param in paramConfig.keys()]).float()

    bounds = np.vstack([lower_bounds, upper_bounds])

    # Create the MinMaxScaler and fit it to the bounds
    scaler = MinMaxScaler().fit(bounds)

    # Transform the parameters using the fitted scaler
    X_normalized = torch.tensor(scaler.transform(X.numpy()), dtype=torch.float64)

    # Standardize Y to have zero mean and unit variance
    Y_standardized = standardize(Y)

    # Normalise the bounds in accordance to the normalised params
    bounds_normalized = torch.tensor([[0.0]*X_normalized.shape[1], [1.0]*X_normalized.shape[1]])

    # Initialize model
    model = SingleTaskGP(X_normalized, Y_standardized)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)

    # Define the acquisition function
    # **Reference Point**

    # qEHVI requires specifying a reference point, which is the lower bound on the objectives used for computing hypervolume. 
    # In this tutorial, we assume the reference point is known. In practice the reference point can be set 
    # 1) using domain knowledge to be slightly worse than the lower bound of objective values, 
    # where the lower bound is the minimum acceptable value of interest for each objective, or 
    # 2) using a dynamic reference point selection strategy.
    ref_point = Y_standardized.max(dim=0).values - 0.01

    partitioning = NondominatedPartitioning(ref_point=ref_point, Y=Y_standardized)
    acq_func = qExpectedHypervolumeImprovement(
        model=model,
        partitioning=partitioning,
        ref_point=ref_point,
        objective=IdentityMCMultiOutputObjective(),
    )

    # Optimize the acquisition function
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds_normalized,
        q=1,#q: This is the number of points to sample in each step
        num_restarts=10,#num_restarts: This is the number of starting points for the optimization.
        raw_samples=1000,#raw_samples: This is the number of samples to draw when initializing the optimization
    )

    # Unnormalize the candidates
    candidates = torch.tensor(scaler.inverse_transform(candidates.detach().numpy()), dtype=torch.float64)

    #converting to dictionary
    pareto_front = [{param: value.item() for param, value in zip(paramConfig.keys(), next_param)} for next_param in candidates]
    return pareto_front

def MOO_calculate_geometries_weight(targetCurves, geometries):
    geometryWeights = {}
    
    for geometry in geometries:
        targetDisplacement = targetCurves[geometry]["displacement"]
        targetForce = targetCurves[geometry]["force"]
        
        x_start = min(targetDisplacement)
        x_end = max(targetDisplacement)

        # Interpolate the force-displacement curve
        target_FD_func = interp1d(targetDisplacement, targetForce, fill_value="extrapolate")

        # Evaluate the two curves at various points within the x-range boundary
        x_values = np.linspace(x_start, x_end, num=10000)

        y_values = target_FD_func(x_values)

        area = simpson(y_values, x_values)
        geometryWeights[geometry] = 1/np.array(area)
    
    # normalize the weights
    sumWeights = np.sum(list(geometryWeights.values()))
    for geometry in geometryWeights:
        geometryWeights[geometry] = geometryWeights[geometry]/sumWeights

    return geometryWeights

def prettyPrint(parameters, paramConfig, logPath):
    logTable = PrettyTable()
    logTable.field_names = ["Parameter", "Value"]
    for param in parameters:
        paramName = paramConfig[param]['name']
        paramValue = parameters[param]
        paramUnit = paramConfig[param]['unit']
        paramValueUnit = f"{paramValue} {paramUnit}" if paramUnit != "dimensionless" else paramValue
        #print(paramName)
        #print(paramValueUnit)
        logTable.add_row([paramName, paramValueUnit])

    stringMessage = "\n"
    stringMessage += logTable.get_string()
    stringMessage += "\n"

    printLog(stringMessage, logPath)

#######################################
# Simulation related helper functions #
#######################################

def read_FD_Curve(filePath):
    output_data=np.loadtxt(filePath, skiprows=2)
    # column 1 is time step
    # column 2 is displacement
    # column 3 is force
    columns=['X', 'Displacement', 'Force']
    df = pd.DataFrame(data=output_data, columns=columns)
    # Converting to numpy array
    displacement = df.iloc[:, 1].to_numpy()
    force = df.iloc[:, 2].to_numpy()
    return displacement, force

def create_parameter_file(filePath, paramsDict):
    columns = ["Parameter", "Value"]
    df = pd.DataFrame(columns=columns)
    for key, value in paramsDict.items():
        df.loc[len(df.index)] = [key, value]
    df.to_excel(f"{filePath}/parameters.xlsx", index=False)
    df.to_csv(f"{filePath}/parameters.csv", index=False)

def create_flowCurve_file(filePath, truePlasticStrain, trueStress):
    columns = ["strain,-", "stress,MPa", "stress,Pa"]
    df = pd.DataFrame(columns=columns)
    for i in range(len(truePlasticStrain)):
        df.loc[len(df.index)] = [truePlasticStrain[i], trueStress[i], trueStress[i]*1e6]
    df.to_excel(f"{filePath}/flowCurve.xlsx", index=False)
    df.to_csv(f"{filePath}/flowCurve.csv", index=False)

def create_FD_Curve_file(filePath, displacement, force):
    columns = ["displacement,mm", "force,kN", "force,N"]
    df = pd.DataFrame(columns=columns)
    for i in range(len(displacement)):
        df.loc[len(df.index)] = [displacement[i], force[i] * 1e-3, force[i]]
    df.to_excel(f"{filePath}/FD_Curve.xlsx", index=False)
    df.to_csv(f"{filePath}/FD_Curve.csv", index=False)

def replace_flowCurve_material_inp(filePath, truePlasticStrain, trueStress):
    with open(filePath, 'r') as material_inp:
        material_inp_content = material_inp.readlines()
    # Locate the section containing the stress-strain data
    start_line = None
    end_line = None
    for i, line in enumerate(material_inp_content):
        if '*Plastic' in line:
            start_line = i + 1
        elif '*Density' in line:
            end_line = i
            break

    if start_line is None or end_line is None:
        raise ValueError('Could not find the stress-strain data section')

    # Modify the stress-strain data
    new_stress_strain_data = zip(trueStress, truePlasticStrain)
    # Update the .inp file
    new_lines = []
    new_lines.extend(material_inp_content[:start_line])
    new_lines.extend([f'{stress},{strain}\n' for stress, strain in new_stress_strain_data])
    new_lines.extend(material_inp_content[end_line:])

    # Write the updated material.inp file
    with open(filePath, 'w') as file:
        file.writelines(new_lines)

def replace_maxDisp_geometry_inp(filePath, maxTargetDisplacement):
    with open(filePath, 'r') as geometry_inp:
        geometry_inp_content = geometry_inp.readlines()
    start_line = None
    end_line = None
    for i, line in enumerate(geometry_inp_content[-60:]):
        if line.startswith('*Boundary, amplitude'):
            original_index = len(geometry_inp_content) - 60 + i
            start_line = original_index + 1
            end_line = original_index + 2
            break

    if start_line is None or end_line is None:
        raise ValueError('Could not find the *Boundary, amplitude displacement section')

    new_disp_data = f"Disp, 2, 2, {maxTargetDisplacement}\n"

    new_lines = []
    new_lines.extend(geometry_inp_content[:start_line])
    new_lines.extend([new_disp_data])
    new_lines.extend(geometry_inp_content[end_line:])

    with open(filePath, 'w') as file:
        file.writelines(new_lines)

def replace_materialName_geometry_inp(filePath, materialName):
    with open(filePath, 'r') as geometry_inp:
        geometry_inp_content = geometry_inp.readlines()
    start_line = None
    end_line = None
    for i, line in enumerate(geometry_inp_content[-60:]):
        if line.startswith('*INCLUDE, INPUT='):
            original_index = len(geometry_inp_content) - 60 + i
            start_line = original_index
            end_line = original_index + 1
            break

    if start_line is None or end_line is None:
        raise ValueError('Could not find the **INCLUDE, INPUT= section')

    new_material_data = f"*INCLUDE, INPUT={materialName}\n"

    new_lines = []
    new_lines.extend(geometry_inp_content[:start_line])
    new_lines.extend([new_material_data])
    new_lines.extend(geometry_inp_content[end_line:])

    with open(filePath, 'w') as file:
        file.writelines(new_lines)

def rescale_paramsDict(paramsDict, paramConfig):
    rescaled_paramsDict = {}
    for param, value in paramsDict.items():
        rescaled_paramsDict[param] = value * paramConfig[param]['exponent']
    return rescaled_paramsDict

def reverseAsParamsToGeometries(curves, geometries):
    exampleGeometry = geometries[0]
    reverseCurves = {}
    for paramsTuple in curves[exampleGeometry]:
        reverseCurves[paramsTuple] = {}
        for geometry in geometries:
            reverseCurves[paramsTuple][geometry] = curves[geometry][paramsTuple]
    return reverseCurves

def calculate_yielding_index(targetDisplacement, targetForce, r2_threshold=0.998):
    """
    This function calculates the end of the elastic (linear) region of the force-displacement curve.
    """
    yielding_index = 0

    # Initialize the Linear Regression model
    linReg = LinearRegression()
    targetDisplacement = np.array(targetDisplacement)
    targetForce = np.array(targetForce)
    for i in range(2, len(targetDisplacement)):
        linReg.fit(targetDisplacement[:i].reshape(-1, 1), targetForce[:i]) 
        simForce = linReg.predict(targetDisplacement[:i].reshape(-1, 1)) 
        r2 = r2_score(targetForce[:i], simForce) 
        if r2 < r2_threshold:  # If R^2 is below threshold, mark the end of linear region
            yielding_index = i - 1
            break
    return yielding_index
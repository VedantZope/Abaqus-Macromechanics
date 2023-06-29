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

def SOO_write_BO_json_log(FD_Curves, targetCurve, paramConfig):
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
        line["target"] = -lossFD(targetCurve["displacement"], targetCurve["force"], dispforce["force"])
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
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

def printLog(message, logPath):
    with open(logPath, 'a+') as logFile:
        logFile.writelines(message)
    print(message)

def parseBoundsBO(paramInfo):
    paramBounds = {}
    for param in paramInfo:
        paramBounds[param] = (paramInfo[param]['lowerBound'], paramInfo[param]['upperBound'])
    return paramBounds

def is_directory_empty(directory_path):
    return len(os.listdir(directory_path)) == 0

def smoothing_force(force):
    start = 30
    end = 60
    smooth_force = copy.deepcopy(force)
    for i in range(20000):
        smooth_force = savgol_filter(smooth_force[start:end], 
                                    window_length=5, 
                                    polyorder=3,
                                    #deriv=0,
                                    #delta=1
                                    )
        smooth_force = np.concatenate((force[0:start], smooth_force, force[end:]))
    return smooth_force

def interpolatingForce(simDisp, simForce, targetDisp):
    interpolatingFunction = interp1d(simDisp, simForce, fill_value='extrapolate')
    # Interpolate the force
    interpolatedSimForce = interpolatingFunction(targetDisp)
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
    return FD_Curves_copy

def SOO_write_BO_json_log(FD_Curves, targetCurve):
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
        line["target"] = -lossFD(targetCurve["force"], dispforce["force"])
        line["params"] = dict(paramsTuple)
        line["datetime"] = {}
        line["datetime"]["datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line["datetime"]["elapsed"] = 0.0
        line["datetime"]["delta"] = 0.0

        # json file has not exist yet
        # Write the dictionary to json file
        with open(f"optimizers/logs.json", "a") as file:
            json.dump(line, file)
            file.write("\n")
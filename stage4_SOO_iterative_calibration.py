import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
from modules.SOO_SIM import *
from modules.MOO_SIM import *
from modules.hardeningLaws import *
from modules.helper import *
from modules.stoploss import *
from optimizers.BO import *
import stage0_configs 
import stage1_prepare_targetCurve
import stage2_run_initialSims 
import stage3_prepare_simCurves
from math import *
import json
from datetime import datetime
import os
import prettytable

def main_SOO_iterative_calibration(info):

    # ---------------------------------------------------#
    #  Stage 4: RUn iterative parameter calibration loop #
    # ---------------------------------------------------#

    
    projectPath = info['projectPath']
    logPath = info['logPath']
    resultPath = info['resultPath']
    simPath = info['simPath']
    targetPath = info['targetPath']
    templatePath = info['templatePath'] 
    material = info['material']
    optimizeStrategy = info['optimizeStrategy']
    optimizerName = info['optimizerName']
    hardeningLaw = info['hardeningLaw']
    paramConfig = info['paramConfig']
    geometry = info['geometry']
    curveIndex = info['curveIndex']
    yielding_deviationPercent = info['yielding_deviationPercent']
    hardening_deviationPercent = info['hardening_deviationPercent']
    numberOfInitialSims = info['numberOfInitialSims']
    template_paramsDict = info['template_paramsDict']
    
    # Continuous searching space
    if optimizerName == "BO":
        param_bounds = parseBoundsBO(info['paramConfig'])
    info['param_bounds'] = param_bounds

    targetCurve = info['targetCurve']

    initial_original_FD_Curves = info['initial_original_FD_Curves']
    iteration_original_FD_Curves = info['iteration_original_FD_Curves']
    combined_original_FD_Curves = info['combined_original_FD_Curves']
    initial_interpolated_FD_Curves = info['initial_interpolated_FD_Curves']
    iteration_interpolated_FD_Curves = info['iteration_interpolated_FD_Curves']
    combined_interpolated_FD_Curves = info['combined_interpolated_FD_Curves']
    initial_original_flowCurves = info['initial_original_flowCurves']
    iteration_original_flowCurves = info['iteration_original_flowCurves']
    combined_original_flowCurves = info['combined_original_flowCurves']
    
    sim = SOO_SIM(info)
    
    lossFD_dict = {"yielding": lossFD_yielding, "hardening": lossFD_hardening}
    stopFD_dict = {"yielding": stopFD_yielding, "hardening": stopFD_hardening}
    yielding_params = [param for param in paramConfig if paramConfig[param]["type"] == "yielding"]
    yielding_params_names = [paramConfig[param]["name"] for param in yielding_params]
    hardening_params = [param for param in paramConfig if paramConfig[param]["type"] == "hardening"]
    hardening_params_names = [paramConfig[param]["name"] for param in hardening_params]
    optimize_params_dict = {"yielding": yielding_params, "hardening": hardening_params}
    print(optimize_params_dict)
    # Filter out the parameters
    paramsConfig_yielding = {param: paramConfig[param] for param in paramConfig if paramConfig[param]["type"] == "yielding"}
    paramsConfig_hardening = {param: paramConfig[param] for param in paramConfig if paramConfig[param]["type"] == "hardening"}
    yielding_params_bounds = parseBoundsBO(paramsConfig_yielding)
    hardening_params_bounds = parseBoundsBO(paramsConfig_hardening)
    param_bounds_dict = {"yielding": yielding_params_bounds, "hardening": hardening_params_bounds}

    deviationPercent_dict = {"yielding": yielding_deviationPercent, "hardening": hardening_deviationPercent}
    
    printLog(f"The yielding parameters are {yielding_params_names}", logPath)
    printLog(f"The hardening parameters are {hardening_params_names}", logPath)

    for stage in ["yielding", "parameter"]:
        printLog("\n" + 60 * "#" + "\n", logPath)
        printLog(f"Starting {stage} parameter calibration stage\n", logPath)
        # Check if the results already exist 
        if os.path.exists(f"{resultPath}/iteration/common/{stage}Stage_FD_Curves.npy"):
            printLog(f"Found existing {stage} stage results", logPath)
            stageResults_FD_Curves = np.load(f"{resultPath}/iteration/common/{stage}Stage_FD_Curves.npy", allow_pickle=True).tolist()
            stageResults_paramsDict = stageResults_FD_Curves["parameters"]
            printLog(f"The {stage} stage parameter result is", logPath)
            prettyPrint(stageResults_paramsDict, optimize_params_dict[stage], paramConfig, logPath)
        else:
            while not stopFD_dict[stage](targetCurve['displacement'], 
                                        targetCurve['force'], 
                                        list(combined_interpolated_FD_Curves.values())[-1]['force'], 
                                        deviationPercent_dict[stage]):
                # Writing the current iteration results to the log file for BO
                
                SOO_write_BO_json_log(stage, optimize_params_dict[stage], combined_interpolated_FD_Curves, targetCurve, paramConfig)
                BO_instance = BO(info)
                BO_instance.initializeOptimizer(lossFunction=None, param_bounds=param_bounds_dict[stage], loadingProgress=True)
                next_paramsDict = BO_instance.suggest()
                next_paramsDict = rescale_paramsDict(next_paramsDict, paramConfig)

                # Assigning the parameter values to the template parameter dictionary
                for param in next_paramsDict:
                    template_paramsDict[param] = next_paramsDict[param]

                # Getting the simulation iteration index. Counting from 1, not 0
                iterationIndex = len(iteration_interpolated_FD_Curves) + 1
                
                printLog("\n" + 60 * "#" + "\n", logPath)
                printLog(f"Running iteration {iterationIndex} for {material}_{hardeningLaw}_{geometry}_curve{curveIndex}" , logPath)
                printLog(f"The next candidate {hardeningLaw} parameters predicted by BO", logPath)
                prettyPrint(template_paramsDict, optimize_params_dict[stage], paramConfig, logPath)

                time.sleep(30)
                printLog("Start running iteration simulation", logPath)
                
                new_FD_Curves, new_flowCurves = sim.run_iteration_simulations(template_paramsDict, iterationIndex)
                
                # Updating the combined FD curves
                combined_original_FD_Curves.update(new_FD_Curves)
                combined_interpolated_FD_Curves = interpolating_FD_Curves(combined_original_FD_Curves, targetCurve)
                
                # Updating the iteration FD curves
                iteration_original_FD_Curves.update(new_FD_Curves)
                iteration_interpolated_FD_Curves = interpolating_FD_Curves(iteration_original_FD_Curves, targetCurve)
                
                # Updating the original flow curves
                combined_original_flowCurves.update(new_flowCurves)
                iteration_original_flowCurves.update(new_flowCurves)

                printLog("Finish saving new iteration data", logPath)
                simForce = list(new_FD_Curves.values())[0]['force']
                loss_newIteration = lossFD_dict[stage](targetCurve['displacement'], targetCurve['force'], simForce)
                printLog(f"The loss of the new iteration is {round(loss_newIteration, 3)}", logPath)

                # Saving the iteration data
                np.save(f"{resultPath}/iteration/common/FD_Curves.npy", iteration_original_FD_Curves)
                np.save(f"{resultPath}/iteration/common/flowCurves.npy", iteration_original_flowCurves)
            
            printLog("\n" + 60 * "#" + "\n", logPath)
            printLog(f"The simulated curve has satisfied the {deviationPercent_dict[stage]} deviation percent of the {stage} stage", logPath)
            
            stageResults_FD_Curves = {"parameters": template_paramsDict, 
                                      "succeeding_iteration": len(iteration_original_FD_Curves),
                                      "displacement": list(iteration_original_FD_Curves.values())[-1]['displacement'],
                                      "force": list(iteration_original_FD_Curves.values())[-1]['force']}
            stageResults_flowCurves = {"parameters": template_paramsDict,
                                        "succeeding_iteration": len(iteration_original_flowCurves),
                                        "strain": list(iteration_original_flowCurves.values())[-1]['strain'],
                                        "stress": list(iteration_original_flowCurves.values())[-1]['stress']}
            np.save(f"{resultPath}/iteration/common/{stage}Stage_FD_Curves.npy", stageResults_FD_Curves) 
            np.save(f"{resultPath}/iteration/common/{stage}Stage_flowCurves.npy", stageResults_flowCurves)
            printLog(f"Saving the {stage} stage results successfully", logPath)
    
        
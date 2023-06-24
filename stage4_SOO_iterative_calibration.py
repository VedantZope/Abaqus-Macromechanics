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

def main_iterative_calibration(info):

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
    deviationPercent = info['deviationPercent']
    numberOfInitialSims = info['numberOfInitialSims']
    
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
    
    if optimizeStrategy == "SOO":
        sim = SOO_SIM(info)
    
    while not stopFD(targetCurve['force'], list(combined_interpolated_FD_Curves.values())[-1]['force'], deviationPercent):

        SOO_write_BO_json_log(combined_interpolated_FD_Curves, targetCurve, paramConfig)
        BO_instance = BO(info)
        BO_instance.initializeOptimizer(lossFunction=None, param_bounds=param_bounds, loadingProgress=True)
        next_paramsDict = BO_instance.suggest()
        next_paramsDict = rescale_paramsDict(next_paramsDict, paramConfig)
        iterationIndex = len(iteration_interpolated_FD_Curves) + 1
        
        printLog("\n" + 60 * "#" + "\n", logPath)
        printLog(f"Running iteration {iterationIndex} for {material}_{hardeningLaw}_{geometry}_curve{curveIndex}" , logPath)
        printLog(f"The next candidate {hardeningLaw} parameters predicted by BO", logPath)
        prettyPrint(next_paramsDict, paramConfig, logPath)

        time.sleep(30)
        printLog("Start running iteration simulation", logPath)
        
        new_FD_Curves, new_flowCurves = sim.run_iteration_simulations(next_paramsDict, iterationIndex)
        
        # Updating the combined FD curves
        combined_original_FD_Curves.update(new_FD_Curves)
        combined_interpolated_FD_Curves = interpolating_FD_Curves(combined_original_FD_Curves, targetCurve)
        
        # Updating the iteration FD curves
        iteration_original_FD_Curves.update(new_FD_Curves)
        iteration_interpolated_FD_Curves = interpolating_FD_Curves(iteration_original_FD_Curves, targetCurve)
        
        # Updating the original flow curves
        combined_original_flowCurves.update(new_flowCurves)
        iteration_original_flowCurves.update(new_flowCurves)

        simForce = list(new_FD_Curves.values())[0]['force']
        loss_newIteration = lossFD(targetCurve['displacement'], targetCurve['force'], simForce)
        printLog(f"The loss of the new iteration is {round(loss_newIteration, 3)}", logPath)

        # Saving the iteration data
        np.save(f"{resultPath}/iteration/common/FD_Curves.npy", iteration_original_FD_Curves)
        np.save(f"{resultPath}/iteration/common/flowCurves.npy", iteration_original_flowCurves)

        # Saving the combined data
        np.save(f"{resultPath}/combined/common/FD_Curves.npy", combined_original_FD_Curves)
        
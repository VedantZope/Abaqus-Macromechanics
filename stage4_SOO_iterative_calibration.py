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
import stage1_SOO_prepare_targetCurve
import stage2_SOO_run_initialSims 
import stage3_SOO_prepare_simCurves
from math import *
import json
from datetime import datetime
import os
import prettytable

def main_iterative_calibration(info):

    # ---------------------------------------------------#
    #  Stage 4: Run iterative parameter calibration loop #
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
    yieldingIndex = info['yieldingIndex']
    
    # Continuous searching space
    param_bounds = parseBoundsBO(info['paramConfig'])
    info['param_bounds'] = param_bounds

    targetCurve = info['targetCurve']

    initial_original_FD_Curves_smooth = info['initial_original_FD_Curves_smooth']
    iteration_original_FD_Curves_smooth = info['iteration_original_FD_Curves_smooth']
    combined_original_FD_Curves_smooth = info['combined_original_FD_Curves_smooth']
    initial_interpolated_FD_Curves_smooth = info['initial_interpolated_FD_Curves_smooth']
    iteration_interpolated_FD_Curves_smooth = info['iteration_interpolated_FD_Curves_smooth']
    combined_interpolated_FD_Curves_smooth = info['combined_interpolated_FD_Curves_smooth']

    iteration_original_FD_Curves_unsmooth = info['iteration_original_FD_Curves_unsmooth']

    initial_original_flowCurves = info['initial_original_flowCurves']
    iteration_original_flowCurves = info['iteration_original_flowCurves']
    combined_original_flowCurves = info['combined_original_flowCurves']
    
    
    sim = SOO_SIM(info)
    
    while not stopFD_SOO(targetCurve['force'], list(combined_interpolated_FD_Curves_smooth.values())[-1]['force'], yieldingIndex, deviationPercent):
        iterationIndex = len(iteration_interpolated_FD_Curves_smooth) + 1
        SOO_write_BO_json_log(combined_interpolated_FD_Curves_smooth, targetCurve, yieldingIndex, paramConfig,iterationIndex)
        BO_instance = BO(info)
        BO_instance.initializeOptimizer(lossFunction=None, param_bounds=param_bounds, loadingProgress=True)
        next_paramsDict = BO_instance.suggest()
        next_paramsDict = rescale_paramsDict(next_paramsDict, paramConfig)
        
        
        #print(len(iteration_interpolated_FD_Curves_smooth))
        printLog("\n" + 60 * "#" + "\n", logPath)
        printLog(f"Running iteration {iterationIndex} for {material}_{hardeningLaw}_{geometry}_curve{curveIndex}" , logPath)
        printLog(f"The next candidate {hardeningLaw} parameters predicted by BO", logPath)
        prettyPrint(next_paramsDict, paramConfig, logPath)

        time.sleep(30)
        printLog("Start running iteration simulation", logPath)
        
        new_FD_Curves, new_flowCurves = sim.run_iteration_simulations(next_paramsDict, iterationIndex)
        new_FD_Curves_unsmooth = copy.deepcopy(new_FD_Curves)
        new_FD_Curves_smooth = copy.deepcopy(new_FD_Curves)
        new_params = list(new_FD_Curves.keys())[0]
        new_FD_Curves_smooth[new_params]['force'] = smoothing_force(new_FD_Curves_unsmooth[new_params]['force'], startIndex=20, endIndex=90, iter=20000)
        
        # Updating the combined FD curves smooth
        combined_original_FD_Curves_smooth.update(new_FD_Curves_smooth)
        combined_interpolated_FD_Curves_smooth = interpolating_FD_Curves(combined_original_FD_Curves_smooth, targetCurve)
        
        # Updating the iteration FD curves smooth
        iteration_original_FD_Curves_smooth.update(new_FD_Curves_smooth)
        iteration_interpolated_FD_Curves_smooth = interpolating_FD_Curves(iteration_original_FD_Curves_smooth, targetCurve)
        
        # Updating the iteration FD curves unsmooth
        iteration_original_FD_Curves_unsmooth.update(new_FD_Curves_unsmooth)
        
        # Updating the original flow curves
        combined_original_flowCurves.update(new_flowCurves)
        iteration_original_flowCurves.update(new_flowCurves)

        simForce = list(new_FD_Curves_smooth.values())[0]['force'][yieldingIndex:]
        simDisplacement = list(new_FD_Curves_smooth.values())[0]['displacement'][yieldingIndex:]
        targetForce = targetCurve['force'][yieldingIndex:]
        targetDisplacement = targetCurve['displacement'][yieldingIndex:]
        interpolated_simForce = interpolatingForce(simDisplacement, simForce, targetDisplacement)
        
        loss_newIteration = lossFD(targetDisplacement, targetForce, interpolated_simForce,iterationIndex)
        printLog(f"The loss of the new iteration is {round(loss_newIteration, 3)}", logPath)

        # Saving the iteration data
        np.save(f"{resultPath}/iteration/common/FD_Curves_unsmooth.npy", iteration_original_FD_Curves_unsmooth)
        np.save(f"{resultPath}/iteration/common/FD_Curves_smooth.npy", iteration_original_FD_Curves_smooth)
        np.save(f"{resultPath}/iteration/common/flowCurves.npy", iteration_original_flowCurves)
        
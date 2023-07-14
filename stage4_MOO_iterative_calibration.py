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
    geometries = info['geometries']
    curveIndex = info['curveIndex']
    deviationPercent = info['deviationPercent']
    numberOfInitialSims = info['numberOfInitialSims']
    targetCurves = info['targetCurves']
    yieldingIndices = info['yieldingIndices']
    
    # Continuous searching space
    if optimizerName == "BO":
        param_bounds = parseBoundsBO(info['paramConfig'])

    initial_original_geom_to_param_FD_Curves_smooth = info['initial_original_geom_to_param_FD_Curves_smooth']
    iteration_original_geom_to_param_FD_Curves_smooth = info['iteration_original_geom_to_param_FD_Curves_smooth']
    combined_original_geom_to_param_FD_Curves_smooth = info['combined_original_geom_to_param_FD_Curves_smooth']
    initial_interpolated_geom_to_param_FD_Curves_smooth = info['initial_interpolated_geom_to_param_FD_Curves_smooth']
    iteration_interpolated_geom_to_param_FD_Curves_smooth = info['iteration_interpolated_geom_to_param_FD_Curves_smooth']
    combined_interpolated_geom_to_param_FD_Curves_smooth = info['combined_interpolated_geom_to_param_FD_Curves_smooth']
    combined_interpolated_param_to_geom_FD_Curves_smooth = info['combined_interpolated_param_to_geom_FD_Curves_smooth']
    iteration_original_geom_to_param_FD_Curves_unsmooth = info['iteration_original_geom_to_param_FD_Curves_unsmooth']
    iteration_original_param_to_geom_FD_Curves_smooth = info['iteration_original_param_to_geom_FD_Curves_smooth']
    
    initial_original_geom_to_param_flowCurves = info['initial_original_geom_to_param_flowCurves']
    iteration_original_geom_to_param_flowCurves = info['iteration_original_geom_to_param_flowCurves']
    combined_original_geom_to_param_flowCurves = info['combined_original_geom_to_param_flowCurves']

    sim = MOO_SIM(info)
    

    
    #np.save("combined_interpolated_param_to_geom_FD_Curves_smooth.npy", combined_interpolated_param_to_geom_FD_Curves_smooth)
    #print("Hello")
    #time.sleep(180)

    # np.save("targetCurves.npy", targetCurves)
    # print("Hello")
    # time.sleep(180)
    while not stopFD_MOO(targetCurves, list(combined_interpolated_param_to_geom_FD_Curves_smooth.values())[-1], geometries, yieldingIndices, deviationPercent):

        iterationIndex = len(iteration_original_param_to_geom_FD_Curves_smooth) + 1

        # Please refer to this one to implement your own optimizer
        # Weighted single objective optimization strategy:
        if optimizerName == "BO":
            geometryWeights = MOO_calculate_geometries_weight(targetCurves, geometries)
            printLog("The weights for the geometries are: ", logPath)
            printLog(str(geometryWeights), logPath)
            exampleGeometry = geometries[0]

            MOO_write_BO_json_log(combined_interpolated_param_to_geom_FD_Curves_smooth, targetCurves, geometries, geometryWeights, yieldingIndices, paramConfig,iterationIndex)
            BO_instance = BO(info)
            BO_instance.initializeOptimizer(lossFunction=None, param_bounds=param_bounds, loadingProgress=True)
            next_paramDict = BO_instance.suggest()
            next_paramDict = rescale_paramsDict(next_paramDict, paramConfig)
            
        # Multiple objective optimization strategy (Vedant's Task)

        # In order to see whats going on, please use the print() plus time.sleep(180) to stop the code to inspect what is inside a variable 
        # For example, if you want to see what is paramConfig, please use the following code:
        # print(paramConfig)
        # print("Hello") -> This print here helps making sure it has stopped at correct place
        # time.sleep(180) 
        # Then run python optimize.py

        # Given these informations, you should be able to program the optimizer to return the next_paramDict
        
        # parameter bounds, exponents and names are stored in paramConfig. Dictionary (paramName) -> dict of parameter properties
        
        # The existing initial and iteration FD curves stored in combined_interpolated_param_to_geom_FD_Curves_smooth
        # combined_interpolated_param_to_geom_FD_Curves_smooth is of structure
        # {paramTuples: {geometryName: {force: np.array(), displacement: np.array()}}}
        
        # where geometryNames are stored in the variable "geometries" of type list
        
        # the targetCurves for storing the experimental FD curves. It is of structure
        # {geometryName: {force: np.array(), displacement: np.array()}}

        # You can also use geometryWeights to weight the geometries in the optimization process if your algorithm needs weights
        # geometryWeights is of structure
        # {geometryName: weight}, where sum of the weight equals to 1

        # Given these information, you should program the optimizer to return the next_paramDict
        # The next_paramDict should be a dictionary with the Swift Voce parameters as keys and 
        # the values are the next predicted parameter values for the next iteration
        # Apart from these, you dont need to care much about anything else
        
        # Please pay attention to the param bounds, where it has not have the exponents. 
        # If your optimizer returns parameters within the bounds, you need to rescale the parameters to the correct scale by multiplying it with the exponents
        
        if optimizerName == "BOTORCH":
            pareto_front = MOO_suggest_BOTORCH(combined_interpolated_param_to_geom_FD_Curves_smooth, targetCurves, geometries, yieldingIndices, paramConfig,iterationIndex)
            # Select a random point from the pareto front
            next_paramDict = pareto_front[0]
        

        #print(len(iteration_interpolated_FD_Curves_smooth))
        printLog("\n" + 60 * "#" + "\n", logPath)
        printLog(f"Running iteration {iterationIndex} for {material}_{hardeningLaw}_curve{curveIndex}" , logPath)
        printLog(f"The next candidate {hardeningLaw} parameters predicted by {optimizerName}", logPath)
        prettyPrint(next_paramDict, paramConfig, logPath)

        time.sleep(30)
        printLog("Start running iteration simulation", logPath)
        
        geom_to_param_new_FD_Curves, geom_to_param_new_flowCurves = sim.run_iteration_simulations(next_paramDict, iterationIndex)
        
        geom_to_param_new_FD_Curves_unsmooth = copy.deepcopy(geom_to_param_new_FD_Curves)
        geom_to_param_new_FD_Curves_smooth = copy.deepcopy(geom_to_param_new_FD_Curves)
        new_param = list(geom_to_param_new_FD_Curves[exampleGeometry].keys())[0]
        
        for geometry in geometries:
            geom_to_param_new_FD_Curves_smooth[geometry][new_param]['force'] = smoothing_force(geom_to_param_new_FD_Curves_unsmooth[geometry][new_param]['force'], startIndex=20, endIndex=90, iter=20000)
        
        # Updating the combined FD curves smooth
        for geometry in geometries:
            combined_original_geom_to_param_FD_Curves_smooth[geometry].update(geom_to_param_new_FD_Curves_smooth[geometry])
            combined_interpolated_geom_to_param_FD_Curves_smooth[geometry] = interpolating_FD_Curves(combined_original_geom_to_param_FD_Curves_smooth[geometry], targetCurves[geometry])
        
        # Updating the iteration FD curves smooth
        for geometry in geometries:
            iteration_original_geom_to_param_FD_Curves_smooth[geometry].update(geom_to_param_new_FD_Curves_smooth[geometry])
            iteration_interpolated_geom_to_param_FD_Curves_smooth[geometry] = interpolating_FD_Curves(iteration_original_geom_to_param_FD_Curves_smooth[geometry], targetCurves[geometry])
        
        # Updating the iteration FD curves unsmooth
        for geometry in geometries:
            iteration_original_geom_to_param_FD_Curves_unsmooth[geometry].update(geom_to_param_new_FD_Curves_unsmooth[geometry])
        
        # Updating the original flow curves
        for geometry in geometries:
            combined_original_geom_to_param_flowCurves[geometry].update(geom_to_param_new_flowCurves[geometry])
            iteration_original_geom_to_param_flowCurves[geometry].update(geom_to_param_new_flowCurves[geometry])
        
        # Updating the param_to_geom data
        combined_interpolated_param_to_geom_FD_Curves_smooth = reverseAsParamsToGeometries(combined_interpolated_geom_to_param_FD_Curves_smooth, geometries)
        iteration_original_param_to_geom_FD_Curves_smooth = reverseAsParamsToGeometries(iteration_original_geom_to_param_FD_Curves_smooth, geometries)

        loss_newIteration = {}
        for geometry in geometries:
            yieldingIndex = yieldingIndices[geometry]
            simForce = list(geom_to_param_new_FD_Curves_smooth[exampleGeometry].values())[0]['force'][yieldingIndex:]
            simDisplacement = list(geom_to_param_new_FD_Curves_smooth[exampleGeometry].values())[0]['displacement'][yieldingIndex:]
            targetForce = targetCurves[geometry]['force'][yieldingIndex:]
            targetDisplacement = targetCurves[geometry]['displacement'][yieldingIndex:]
            interpolated_simForce = interpolatingForce(simDisplacement, simForce, targetDisplacement)
            loss_newIteration[geometry] = round(lossFD(targetDisplacement, targetForce, interpolated_simForce,iterationIndex), 3)
        
        printLog(f"The loss of the new iteration is: ", logPath)
        printLog(str(loss_newIteration), logPath)

        # Saving the iteration data
        for geometry in geometries:
            np.save(f"{resultPath}/{geometry}/iteration/common/FD_Curves_unsmooth.npy", iteration_original_geom_to_param_FD_Curves_unsmooth[geometry])
            np.save(f"{resultPath}/{geometry}/iteration/common/FD_Curves_smooth.npy", iteration_original_geom_to_param_FD_Curves_smooth[geometry])
            np.save(f"{resultPath}/{geometry}/iteration/common/flowCurves.npy", iteration_original_geom_to_param_flowCurves[geometry])
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
from modules.SIM import *
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

    while not stopFD(targetCurve['force'], list(combined_interpolated_FD_Curves.values())[-1]['force'], deviationPercent):
        SOO_write_BO_json_log(combined_interpolated_FD_Curves, targetCurve, paramConfig)
        BO_instance = BO(info)
        BO_instance.initializeOptimizer(lossFunction=None, param_bounds=param_bounds, loadingProgress=True)
        next_params = BO_instance.suggest()
        printLog(f"The next candidate {hardeningLaw} parameters predicted by BO", logPath)
        prettyPrint(next_params, paramConfig, logPath)
        
        print("Hello")
        time.sleep(180)

        #solution_dict, solution_tuple, best_solution_loss = BO_instance.outputResult()

        #for param in solution_dict:
        #    print(f"{param}: {solution_dict[param]}")

    # plt.plot(expt_Disp,expt_Force)
    # plt.plot(sim_Disp,sim_Force)
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
from modules.SOO_SIM import *
from modules.hardeningLaws import *
from modules.helper import *
from modules.stoploss import *
from optimizers.BO import *
import stage0_configs 
import stage1_SOO_prepare_targetCurve
import stage2_SOO_run_initialSims 
from math import *
import json
from datetime import datetime
import os
import prettytable
import copy

def main_prepare_simCurves(info):

    # -------------------------------------------------------------------------------------#
    #   Step 3: Preparing FD curves and flow curves from initial and iteration simulations #
    # -------------------------------------------------------------------------------------#
    
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
    
    targetCurve = info['targetCurve']
    
    # Loading initial simulations
    initial_original_FD_Curves_unsmooth = np.load(f"{resultPath}/initial/common/FD_Curves_unsmooth.npy", allow_pickle=True).tolist()
    initial_original_FD_Curves_smooth = np.load(f"{resultPath}/initial/common/FD_Curves_smooth.npy", allow_pickle=True).tolist()
    initial_original_flowCurves = np.load(f"{resultPath}/initial/common/flowCurves.npy", allow_pickle=True).tolist()
    
    # Check if there are any iteration simulations
    if not os.path.exists(f"{resultPath}/iteration/common/FD_Curves_unsmooth.npy"):
        printLog("There are no iteration simulations. Program starts running the iteration simulations", logPath)
        iteration_original_FD_Curves_smooth = {}
        iteration_original_FD_Curves_unsmooth = {}
        iteration_original_flowCurves = {}
    else:
        printLog("Iteration simulations exist", logPath)
        numberOfIterationSims = len(np.load(f"{resultPath}/iteration/common/FD_Curves_unsmooth.npy", allow_pickle=True).tolist())
        printLog(f"Number of iteration simulations: {numberOfIterationSims} FD curves", logPath)
        iteration_original_FD_Curves_smooth = np.load(f"{resultPath}/iteration/common/FD_Curves_smooth.npy", allow_pickle=True).tolist()
        iteration_original_FD_Curves_unsmooth = np.load(f"{resultPath}/iteration/common/FD_Curves_unsmooth.npy", allow_pickle=True).tolist()
        iteration_original_flowCurves = np.load(f"{resultPath}/iteration/common/flowCurves.npy", allow_pickle=True).tolist()

    combined_original_FD_Curves_smooth = copy.deepcopy(initial_original_FD_Curves_smooth)
    combined_original_FD_Curves_smooth.update(iteration_original_FD_Curves_smooth)
    combined_original_flowCurves = copy.deepcopy(initial_original_flowCurves)
    combined_original_flowCurves.update(iteration_original_flowCurves)

    initial_interpolated_FD_Curves_smooth = interpolating_FD_Curves(initial_original_FD_Curves_smooth, targetCurve)
    iteration_interpolated_FD_Curves_smooth = interpolating_FD_Curves(iteration_original_FD_Curves_smooth, targetCurve)
    combined_interpolated_FD_Curves_smooth = interpolating_FD_Curves(combined_original_FD_Curves_smooth, targetCurve)

    FD_Curves_dict = {}
    flowCurves_dict = {}

    FD_Curves_dict['initial_original_FD_Curves_smooth'] = initial_original_FD_Curves_smooth
    FD_Curves_dict['iteration_original_FD_Curves_smooth'] = iteration_original_FD_Curves_smooth
    FD_Curves_dict['combined_original_FD_Curves_smooth'] = combined_original_FD_Curves_smooth
    FD_Curves_dict['initial_interpolated_FD_Curves_smooth'] = initial_interpolated_FD_Curves_smooth
    FD_Curves_dict['iteration_interpolated_FD_Curves_smooth'] = iteration_interpolated_FD_Curves_smooth
    FD_Curves_dict['combined_interpolated_FD_Curves_smooth'] = combined_interpolated_FD_Curves_smooth
    
    FD_Curves_dict['iteration_original_FD_Curves_unsmooth'] = iteration_original_FD_Curves_unsmooth
    
    flowCurves_dict['initial_original_flowCurves'] = initial_original_flowCurves
    flowCurves_dict['iteration_original_flowCurves'] = iteration_original_flowCurves
    flowCurves_dict['combined_original_flowCurves'] = combined_original_flowCurves

    return FD_Curves_dict, flowCurves_dict

    
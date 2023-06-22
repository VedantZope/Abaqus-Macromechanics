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
import stage1_prepare_targetCurve
import stage2_run_initialSims 
import stage3_prepare_simCurves
import stage4_iterative_calibration
from math import *
import json
from datetime import datetime
import os
import prettytable

def main_optimize():

    # -------------------------------#
    #  Automated optimization stages #
    # -------------------------------#

    info = stage0_configs.main_config()
    
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
    
    targetCurve, maxTargetDisplacement = stage1_prepare_targetCurve.main_prepare_targetCurve(info)
    info['targetCurve'] = targetCurve
    info['maxTargetDisplacement'] = maxTargetDisplacement

    stage2_run_initialSims.main_run_initialSims(info)

    FD_Curves_dict, flowCurves_dict = stage3_prepare_simCurves.main_prepare_simCurves(info) 
    info["initial_original_FD_Curves"] = FD_Curves_dict['initial_original_FD_Curves']
    info["iteration_original_FD_Curves"] = FD_Curves_dict['iteration_original_FD_Curves']
    info["combined_original_FD_Curves"] = FD_Curves_dict['combined_original_FD_Curves']
    info["initial_interpolated_FD_Curves"] = FD_Curves_dict['initial_interpolated_FD_Curves']
    info["iteration_interpolated_FD_Curves"] = FD_Curves_dict['iteration_interpolated_FD_Curves']
    info["combined_interpolated_FD_Curves"] = FD_Curves_dict['combined_interpolated_FD_Curves']
    info["initial_original_flowCurves"] = flowCurves_dict['initial_original_flowCurves']
    info["iteration_original_flowCurves"] = flowCurves_dict['iteration_original_flowCurves']
    info["combined_original_flowCurves"] = flowCurves_dict['combined_original_flowCurves']

    stage4_iterative_calibration.main_iterative_calibration(info)
    
if __name__ == "__main__":
    main_optimize()
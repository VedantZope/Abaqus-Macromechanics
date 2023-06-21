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
from stage0_configs import * 
from stage1_prepare_targetCurve import *
from math import *
import json
from datetime import datetime
import os
import prettytable

def main_run_initialSims(info):

    # ---------------------------------------#
    #   Step 2: Running initial simulations  #
    # ---------------------------------------#
    
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
    
    sim = SIM(info)
    if not os.path.exists(f"{resultPath}/initial/common/FD_Curves.npy"):
        printLog("There are no initial simulations. Program starts running the initial simulations", logPath)
        if optimizeStrategy == "SOO":
            sim.SOO_run_initial_simulations()
        elif optimizeStrategy == "MOO":
            sim.MOO_run_initial_simulations()
    else: 
        printLog("Initial simulations already exist", logPath)
        numberOfInitialSims = len(np.load(f"{resultPath}/initial/common/FD_Curves.npy", allow_pickle=True).tolist())
        printLog(f"Number of initial simulations: {numberOfInitialSims} FD curves", logPath)
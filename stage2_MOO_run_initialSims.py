import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
from modules.MOO_SIM import *
from modules.hardeningLaws import *
from modules.helper import *
from modules.stoploss import *
from optimizers.BO import *
from stage0_configs import * 
from stage1_MOO_prepare_targetCurve import *
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
    geometries = info['geometries']
    deviationPercent = info['deviationPercent']
    numberOfInitialSims = info['numberOfInitialSims']
    generateParams = info['generateParams'] 
    maxTargetDisplacements = info['maxTargetDisplacements']


    if generateParams == "manual":
        parameters = np.load(f"{resultPath}/parameters.npy", allow_pickle=True).tolist()
        info['numberOfInitialSims'] = len(parameters)

    elif generateParams == "auto":
        parameters = sim.latin_hypercube_sampling(geometry)

    for geometry in geometries:
        infoCopy = copy.deepcopy(info)
        resultPathGeometry = f"{resultPath}/{geometry}"
        simPathGeometry = f"{simPath}/{geometry}"
        templatePathGeometry = f"{templatePath}/{geometry}"
        infoCopy['resultPath'] = resultPathGeometry
        infoCopy['simPath'] = simPathGeometry
        infoCopy['templatePath'] = templatePathGeometry
        infoCopy['maxTargetDisplacement'] = maxTargetDisplacements[geometry]
        sim = MOO_SIM(infoCopy) 

        if not os.path.exists(f"{resultPathGeometry}/initial/common/FD_Curves_unsmooth.npy"):
            printLog("=======================================================================", logPath)
            printLog(f"There are no initial simulations for {geometry} geometry", logPath)
            printLog(f"Program starts running the initial simulations for {geometry} geometry", logPath)
            sim.run_initial_simulations(parameters)
            printLog(f"Initial simulations for {geometry} geometry have completed", logPath)
        else: 
            printLog("=======================================================================", logPath)
            printLog(f"Initial simulations for {geometry} geometry already exist", logPath)
            numberOfInitialSims = len(np.load(f"{resultPathGeometry}/initial/common/FD_Curves_unsmooth.npy", allow_pickle=True).tolist())
            printLog(f"Number of initial simulations for {geometry} geometry: {numberOfInitialSims} FD curves", logPath)
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
from stage0_configs import * 
from math import *
import json
from datetime import datetime
import os
import prettytable

def main_prepare_targetCurve(info):

    # ----------------------------------#
    #   Step 1: Preparing target curve  #
    # ----------------------------------#
    
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
    

    targetCurves = {}
    maxTargetDisplacements = {}
    for geometry in geometries:
        df = pd.read_csv(f'{targetPath}/{geometry}/FD_Curve.csv')
        expDisplacement = df['displacement/mm'].to_numpy()
        expForce = df['force/N'].to_numpy()
        targetCurve = {}
        targetCurve['displacement'] = expDisplacement
        targetCurve['force'] = expForce
        maxTargetDisplacement = ceil(max(expDisplacement) * 10) / 10
        targetCurves[geometry] = targetCurve
        maxTargetDisplacements[geometry] = maxTargetDisplacement
    # print(targetCurves)
    # print(maxTargetDisplacements)
    # time.sleep(30)
    return targetCurves, maxTargetDisplacements


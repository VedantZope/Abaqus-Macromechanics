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
from math import *
import json
from datetime import datetime
import os

def main_optimize():

    # -------------------------------------------------------------------#
    #   Step 0: Running initial simulations                              #
    # -------------------------------------------------------------------#

    info = main_config()
    
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
    

    # Read the CSV target curve file into a DataFrame (ground truth)
    df = pd.read_csv(f'{targetPath}/FD_Curve.csv')
    expDisplacement = df['displacement/mm'].to_numpy()
    expForce = df['force/N'].to_numpy()
    targetCurve = {}
    targetCurve['displacement'] = expDisplacement
    #targetCurve['force'] = smoothing_force(expForce)
    maxTargetDisplacement = ceil(max(expDisplacement) * 10) / 10
    info['targetCurve'] = targetCurve
    info['maxTargetDisplacement'] = maxTargetDisplacement
    #print(maxTargetDisplacement)
    #time.sleep(30)

    sim = SIM(info)
    if not os.path.exists(f"{resultPath}/initial/common/FD_Curves.npy"):
        printLog("There are no initial simulations. Program starts running the initial simulations.", logPath)
        if optimizeStrategy == "SOO":
            sim.SOO_run_initial_simulations()
        elif optimizeStrategy == "MOO":
            sim.MOO_run_initial_simulations()
    else: 
        printLog("Initial simulations already exist", logPath)
        numberOfInitialSims = len(np.load(f"{resultPath}/initial/common/FD_Curves.npy", allow_pickle=True).tolist())
        printLog(f"Number of initial simulations: {numberOfInitialSims} FD curves", logPath)
    
    FD_Curves = np.load(f"{resultPath}/initial/common/FD_Curves.npy", allow_pickle=True).tolist()
    #print(FD_Curves)
    flowCurves = np.load(f"{resultPath}/initial/common/flowCurves.npy", allow_pickle=True).tolist()
    #print(flowCurves)
    #time.sleep(180)

    # Continuous searching space
    if optimizerName == "BO":
        param_bounds = parseBoundsBO(info['paramConfig'])
    info['param_bounds'] = param_bounds
    #print(param_bounds)

    FD_Curves = interpolating_FD_Curves(FD_Curves, targetCurve)
    #print(FD_Curves)
    #time.sleep(180)
    
    SOO_write_BO_json_log(FD_Curves, targetCurve)
    print("Hello")
    time.sleep(180)
    
    while not stopFD(targetCurve['force'], list(FD_Curves.values())[-1]['force'], deviationPercent):
        BO_instance = BO()
        BO_instance.initializeOptimizerWithoutLossFunction(param_bounds)
        next_param = BO_instance.suggest()
        print(next_param)
        time.sleep(180)
        #solution_dict, solution_tuple, best_solution_loss = BO_instance.outputResult()

        #for param in solution_dict:
        #    print(f"{param}: {solution_dict[param]}")

    # plt.plot(expt_Disp,expt_Force)
    # plt.plot(sim_Disp,sim_Force)

if __name__ == "__main__":
    main_optimize()
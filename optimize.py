import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from scipy.interpolate import CubicSpline
from modules.SIM import *
from modules.hardeningLaws import *
from modules.helper import *
from modules.stoploss import *
from optimizers.BO import *
from stage0_configs import * 


def main_optimize():

    # -------------------------------------------------------------------
    #   Step 0: Running initial simulations
    # -------------------------------------------------------------------

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
    runInitialSims = info['runInitialSims']
    numberOfInitialSims = info['numberOfInitialSims']
    strainStart = info['strainStart']
    strainEnd = info['strainEnd']
    strainStep = info['strainStep']

    sim = SIM(info)
    if runInitialSims == "yes":
        sim.run_initial_simulations()
    # Read the CSV file into a DataFrame (ground truth)
    df = pd.read_csv(f'{targetPath}/{geometry}/Force-Displacement.csv')
    expDisp = df['Disp/mm'].to_numpy()
    expForce = df['Force/kN'].to_numpy()
    info['expDisp'] = expDisp
    info['expForce'] = expForce

    # Continuous searching space
    if optimizerName == "BO":
        param_bounds = parseBoundsBO(info['paramConfig'])
    info['param_bounds'] = param_bounds
    #print(param_bounds)
    time.sleep(80)
    
   
    
    # Note: BO in Bayes-Opt tries to maximize, so you should use the negative of the loss function.
    def lossFunction(**solution):
        #print(solution)
        
         # Adding a jitter to the ending strain to include the last point
        truePlasticStrain = np.arange(strainStart, strainEnd + 1e-10, strainStep)
        if hardeningLaw == "Swift":
            c1, c2, c3 = solution["c1"], solution["c2"], solution["c3"]
            trueStress = Swift(c1, c2, c3, truePlasticStrain)
        elif hardeningLaw == "SwiftVoce":
            c1, c2, c3, c4, c5, c6, c7 = solution["c1"], solution["c2"], solution["c3"], solution["c4"], solution["c5"], solution["c6"]
            trueStress = SwiftVoce(c1, c2, c3, c4, c5, c6, c7, truePlasticStrain)
        
        #===========creating the material input data usin 3 params swift equation============
        
        sim_Disp,sim_Force = sim.run_iteration_simulation(trueStress, trueStrain)
        sim_Disp = np.array(sim_Disp)
        sim_Force = np.array(sim_Force)
        # Sort simulated data by displacement (if not sorted already)
        sort_indices = np.argsort(sim_Disp)
        sim_Disp = sim_Disp[sort_indices]
        sim_Force = sim_Force[sort_indices]

        # Create a cubic spline interpolation function
        cubic_spline = CubicSpline(sim_Disp, sim_Force)

        # Evaluate the interpolated function at the x values of the experimental data
        interpolated_simForce = cubic_spline(expDisp)

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(expForce, interpolated_simForce))
        return -rmse
    
    while not stopFD():
        BO_instance = BO()
        BO_instance.initializeOptimizer(lossFunction, param_bounds)
        BO_instance.run()
        solution_dict, solution_tuple, best_solution_loss = BO_instance.outputResult()

        for param in solution_dict:
            print(f"{param}: {solution_dict[param]}")

    # plt.plot(expt_Disp,expt_Force)
    # plt.plot(sim_Disp,sim_Force)

if __name__ == "__main__":
    main_optimize()
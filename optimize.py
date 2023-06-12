import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from scipy.interpolate import CubicSpline
from modules.SIM import *
from optimizers.BO import *
from step0_configs import * 

def main_optimize():

    # -------------------------------------------------------------------
    #   Step 0: Running initial simulations
    # -------------------------------------------------------------------

    info = stage1_initial_simulations.main_initialSims(info)

    # -------------------------------------------------------------------
    #   Step 1: Extracting the experimental and simulated data
    # -------------------------------------------------------------------

    prepared_data = stage2_prepare_data.main_prepareData(info)

    # -------------------------------------------------------------------
    #   Step 2: Training the ANN models
    # -------------------------------------------------------------------

    if optimizeStrategy.endswith('withOptimizer'):
        trained_models = stage3_trainRegressor_withOptimizer.main_trainRegressor_withOptimizer(info, prepared_data, logging=True)
    else:
        #trained_models = stage2_train_ANN_withoutOptimizer.main_trainANN_withoutOptimizer(info, prepared_data, logging=True)
        trained_models = stage3_trainRegressor_withoutOptimizer.main_trainRegressor_withoutOptimizer(info, prepared_data, logging=True)
    # -------------------------------------------------------------------
    #   Step 3: Analyzing the optimization stages
    # -------------------------------------------------------------------

    stages_data = stage4_stages_analysis.main_stagesAnalysis(info, prepared_data)

    # -------------------------------------------------------------------
    #   Step 4: Optimize the parameters for the curves
    # -------------------------------------------------------------------        

    if optimizeStrategy.endswith("withOptimizer"):
        stage5_withOptimizer.main_withOptimizer(info, prepared_data, stages_data, trained_models)
    elif optimizeStrategy == "withoutOptimizer":
        stage5_withoutOptimizer.main_withoutOptimizer(info, prepared_data, stages_data, trained_models)
    # Outside the for-loop of 4 optimization stages

    # ------------------------------
    #   Finalizing the optimization 
    # ------------------------------

    printLog("\n" + 70 * "=" + "\n", logPath)
    printLog(f"All three optimization stages have successfully completed for curve {CPLaw}{curveIndex}\n", logPath)
    printLog("Congratulations! Thank you for using the Crystal Plasticity Software\n", logPath)
    time.sleep(30)
    
    # Read the CSV file into a DataFrame (ground truth)
    df = pd.read_csv('Disp-Force_ExpRT_ndb50.csv')
    # print(df)
    # Extract the true strain and average true stress columns
    expt_Disp = df['Disp /mm'].to_numpy()
    expt_Force = df['Force /kN'].to_numpy()

    # Continuous searching space
    param_bounds = {
        "c1": (700, 1800),  
        "c2": (0.1 * 1e-14, 10 * 1e-14) ,    
        "c3": (0.001 , 0.1 ) 
    }

    # Note: BO in Bayes-Opt tries to maximize, so you should use the inverse of the loss function.
    def lossFunction( **solution):
        #print(solution)
        c1 = solution["c1"]
        c2 = solution["c2"] 
        c3 = solution["c3"]
        sim_Disp,sim_Force = sim.get_xy(c1,c2,c3)
        sim_Disp = np.array(sim_Disp)
        sim_Force = np.array(sim_Force)
        # Sort simulated data by displacement (if not sorted already)
        sort_indices = np.argsort(sim_Disp)
        sim_Disp = sim_Disp[sort_indices]
        sim_Force = sim_Force[sort_indices]

        # Create a cubic spline interpolation function
        cubic_spline = CubicSpline(sim_Disp, sim_Force)

        # Evaluate the interpolated function at the x values of the experimental data
        interp_sim_Force = cubic_spline(expt_Disp)

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(expt_Force, interp_sim_Force))
        return -rmse

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
# External libraries
import os
import numpy as np
import stage0_config as stage0_config
from modules.SIM_damask2 import *
from modules.preprocessing import *
from modules.helper import *
from prettytable import PrettyTable
    # -------------------------------------------------------------------
    #   Step 1: Loading progress and preparing data
    # -------------------------------------------------------------------

def main_prepareData(info):
    logPath = info['logPath']
    initialResultPath = info['initialResultPath']
    iterationResultPath = info['iterationResultPath']
    initialSimPath = info['initialSimPath']
    iterationSimPath = info['iterationSimPath']
    targetPath = info["targetPath"]
    templatePath = info["templatePath"]
    paramInfoPath = info["paramInfoPath"]
    server = info['server']
    loadings = info['loadings']
    CPLaw = info['CPLaw']
    convertUnit = info['convertUnit']
    initialSims = info['initialSims']
    curveIndex = info['curveIndex']
    projectPath = info['projectPath']
    optimizerName = info['optimizerName']
    param_info = info['param_info']
    material = info['material']
    method = info['method']
    searchingSpace = info['searchingSpace']
    roundContinuousDecimals = info['roundContinuousDecimals']
    loadings = info['loadings']
    exampleLoading = info['exampleLoading']
    yieldingPoints = info['yieldingPoints']
    weightsYieldingConstitutive = info['weightsYieldingConstitutive']
    weightsHardeningConstitutive = info['weightsHardeningConstitutive']
    weightsYieldingLinearLoadings = info['weightsYieldingLinearLoadings']
    weightsHardeningLinearLoadings = info['weightsHardeningLinearLoadings']
    weightsHardeningAllLoadings = info['weightsHardeningAllLoadings']
    paramsFormatted = info['paramsFormatted']
    paramsUnit = info['paramsUnit']
    linearYieldingDev = info['linearYieldingDev']

    printLog("\n" + 70 * "*" + "\n\n", logPath)
    printLog(f"Step 1: Loading progress and preparing data for curve {CPLaw}{curveIndex}\n\n", logPath)

    # Extracting the initial simulation data
    
    initial_loadings_processCurves = {}
    initial_loadings_trueCurves = {}

    for loading in loadings:
        initial_loadings_trueCurves[loading] = np.load(f'{initialResultPath}/{loading}/initial_processCurves.npy', allow_pickle=True).tolist()
        initial_loadings_processCurves[loading] = np.load(f'{initialResultPath}/{loading}/initial_trueCurves.npy', allow_pickle=True).tolist()
        #print(initial_loadings_trueCurves[loading])
        #time.sleep(180)
        #print(list(initial_loadings_trueCurves[loading].keys())[0])
    #time.sleep(180)
    # # Calculating average strain from initial simulations 
    average_initialStrains = {}
    for loading in loadings:
        average_initialStrains[loading] = np.array(list(map(lambda strainstress: strainstress["strain"], initial_loadings_processCurves[loading].values()))).mean(axis=0)
    
    # Producing all target curves npy file
    getTargetCurves(info, Pa=False)

    # Preparing the experimental curve of all loadings
    exp_curve = {}
    exp_curve["true"] = {}
    exp_curve["process"] = {}
    exp_curve["interpolate"] = {}

    # Loading the target curve, calculating the interpolating curve and save the compact data of target curve
    for loading in loadings:
        example_sim_stress = list(initial_loadings_processCurves[loading].values())[0]["stress"]
        exp_trueCurve = np.load(f'{targetPath}/{loading}/{CPLaw}{curveIndex}_true.npy', allow_pickle=True).tolist()
        exp_processCurve = np.load(f'{targetPath}/{loading}/{CPLaw}{curveIndex}_process.npy', allow_pickle=True).tolist()
        interpolatedStrain = interpolatingStrain(average_initialStrains[loading], exp_processCurve["strain"], example_sim_stress, yieldingPoints[CPLaw][loading], loading)                 
        #print(interpolatedStrain)
        #print(exp_processCurve["strain"])
        #print(exp_processCurve["stress"])
        interpolatedStress = interpolatingStress(exp_processCurve["strain"], exp_processCurve["stress"], interpolatedStrain, loading).reshape(-1)
        exp_interpolateCurve = {
            "strain": interpolatedStrain,
            "stress": interpolatedStress
        }
        exp_curve["true"][loading] = exp_trueCurve
        exp_curve["process"][loading] = exp_processCurve
        exp_curve["interpolate"][loading] = exp_interpolateCurve 
        #print(exp_curve["interpolate"][loading])

        np.save(f"targets/{material}/{CPLaw}/{loading}/{CPLaw}{curveIndex}_interpolate.npy", exp_curve["interpolate"][loading])
    stacked_exp_stress = np.hstack(list(map(lambda loading: exp_curve["interpolate"][loading]["stress"], loadings)))
    #print(stacked_exp_stress)
    #print(stacked_exp_stress.size)
    #time.sleep(180)
    np.save(f"targets/{material}/{CPLaw}/common/{CPLaw}{curveIndex}_curves.npy", exp_curve)
    printLog(f"Finished preparing all target curves\n\n", logPath)
    ##################################################################

    #print(exp_curve)
    #time.sleep(30)
  
    # time.sleep(30)
    # Loading iteration curves
    iteration_loadings_trueCurves = {}
    iteration_loadings_processCurves = {}

    for loading in loadings:
        iteration_loadings_trueCurves[loading] = {}
        iteration_loadings_processCurves[loading] = {}

    for loading in loadings:
        if os.path.exists(f"{iterationResultPath}/{loading}/iteration_processCurves.npy"):
            # Loading iteration curves
            iteration_loadings_trueCurves[loading] = np.load(f'{iterationResultPath}/{loading}/iteration_trueCurves.npy', allow_pickle=True).tolist()
            iteration_loadings_processCurves[loading] = np.load(f'{iterationResultPath}/{loading}/iteration_processCurves.npy', allow_pickle=True).tolist()
        else:
            for loading in loadings:
                iteration_loadings_trueCurves[loading] = {}
                iteration_loadings_processCurves[loading] = {}
        
    if os.path.exists(f"{iterationResultPath}/common/stage_CurvesList.npy"):
        # Iteration curves info
        stage_CurvesList = np.load(f'{iterationResultPath}/common/stage_CurvesList.npy', allow_pickle=True).tolist()
    else:
        stage_CurvesList = []




    # Create combined curves
    combined_loadings_trueCurves = {}
    combined_loadings_processCurves = {}
    
    # Updating the combine curves with the initial simulations and iteration curves 
    
    for loading in loadings:
        combined_loadings_trueCurves[loading] = {}
        combined_loadings_processCurves[loading] = {}
        
        combined_loadings_trueCurves[loading].update(initial_loadings_trueCurves[loading])
        combined_loadings_processCurves[loading].update(initial_loadings_processCurves[loading])

        combined_loadings_trueCurves[loading].update(iteration_loadings_trueCurves[loading])
        combined_loadings_processCurves[loading].update(iteration_loadings_processCurves[loading])
    

    initial_loadings_interpolateCurves = {}
    iteration_loadings_interpolateCurves = {}

    # Calculating the interpolated curves from combine curves and derive reverse_interpolate curves
 
    for loading in loadings:
        initial_loadings_interpolateCurves[loading] = {}
        for paramsTuple in initial_loadings_processCurves[loading]:
            sim_strain = initial_loadings_processCurves[loading][paramsTuple]["strain"]
            sim_stress = initial_loadings_processCurves[loading][paramsTuple]["stress"]
            initial_loadings_interpolateCurves[loading][paramsTuple] = {}
            initial_loadings_interpolateCurves[loading][paramsTuple]["strain"] = exp_curve["interpolate"][loading]["strain"] 
            initial_loadings_interpolateCurves[loading][paramsTuple]["stress"] = interpolatingStress(sim_strain, sim_stress, exp_curve["interpolate"][loading]["strain"], loading).reshape(-1) 

    for loading in loadings:
        iteration_loadings_interpolateCurves[loading] = {}
        for paramsTuple in iteration_loadings_processCurves[loading]:
            sim_strain = iteration_loadings_processCurves[loading][paramsTuple]["strain"]
            sim_stress = iteration_loadings_processCurves[loading][paramsTuple]["stress"]
            iteration_loadings_interpolateCurves[loading][paramsTuple] = {}
            iteration_loadings_interpolateCurves[loading][paramsTuple]["strain"] = exp_curve["interpolate"][loading]["strain"] 
            iteration_loadings_interpolateCurves[loading][paramsTuple]["stress"] = interpolatingStress(sim_strain, sim_stress, exp_curve["interpolate"][loading]["strain"], loading).reshape(-1) 
    
    
    
    # Updating the combine curves with the initial simulations and iteration curves 
    combined_loadings_interpolateCurves = {}
    for loading in loadings:
        combined_loadings_interpolateCurves[loading] = {}
        combined_loadings_interpolateCurves[loading].update(initial_loadings_interpolateCurves[loading])
        combined_loadings_interpolateCurves[loading].update(iteration_loadings_interpolateCurves[loading])

    reverse_initial_loadings_trueCurves = reverseAsParamsToLoading(initial_loadings_trueCurves, loadings, exampleLoading)
    reverse_initial_loadings_processCurves = reverseAsParamsToLoading(initial_loadings_processCurves, loadings, exampleLoading)
    reverse_initial_loadings_interpolateCurves = reverseAsParamsToLoading(initial_loadings_interpolateCurves, loadings, exampleLoading)

    reverse_iteration_loadings_trueCurves = reverseAsParamsToLoading(iteration_loadings_trueCurves, loadings, exampleLoading)
    reverse_iteration_loadings_processCurves = reverseAsParamsToLoading(iteration_loadings_processCurves, loadings, exampleLoading)
    reverse_iteration_loadings_interpolateCurves = reverseAsParamsToLoading(iteration_loadings_interpolateCurves, loadings, exampleLoading)

    reverse_combined_loadings_trueCurves = reverseAsParamsToLoading(combined_loadings_trueCurves, loadings, exampleLoading)
    reverse_combined_loadings_processCurves = reverseAsParamsToLoading(combined_loadings_processCurves, loadings, exampleLoading)
    reverse_combined_loadings_interpolateCurves = reverseAsParamsToLoading(combined_loadings_interpolateCurves, loadings, exampleLoading)

    ##########################

    printLog(f"Curve {CPLaw}{curveIndex} info: \n", logPath)

    logTable = PrettyTable()

    logTable.field_names = ["Loading", "Exp yield stress ", "Allowed yield stress sim range"]
    for loading in loadings:
        if loading.startswith("linear"):
            targetYieldStress = '{:.3f}'.format(round(exp_curve["interpolate"][loading]['stress'][1], 3)) 
            rangeSimYieldBelow = '{:.3f}'.format(round(exp_curve["interpolate"][loading]["stress"][1] * (1 - linearYieldingDev * 0.01), 3)) 
            rangeSimYieldAbove = '{:.3f}'.format(round(exp_curve["interpolate"][loading]['stress'][1] * (1 + linearYieldingDev * 0.01), 3)) 
            logTable.add_row([loading, f"{targetYieldStress} MPa", f"[{rangeSimYieldBelow}, {rangeSimYieldAbove} MPa]"])
    
    printLog(logTable.get_string() + "\n\n", logPath)



    # Length of initial and iteration simulations
    initial_length = len(reverse_initial_loadings_processCurves)
    iteration_length = len(reverse_iteration_loadings_processCurves)

    printLog(f"Curve {CPLaw}{curveIndex} status: \n", logPath)
    printLog(f"{iteration_length} iteration simulations completed\n", logPath)
    printLog(f"{initial_length} initial simulations completed\n", logPath)     
    printLog(f"Total: {initial_length + iteration_length} simulations completed\n", logPath)
    printLog(f"Experimental and simulated curves preparation for {CPLaw}{curveIndex} has completed\n", logPath)

    np.save(f'{initialResultPath}/common/initial_loadings_trueCurves.npy', initial_loadings_trueCurves) 
    np.save(f'{initialResultPath}/common/initial_loadings_processCurves.npy', initial_loadings_processCurves)
    np.save(f'{initialResultPath}/common/initial_loadings_interpolateCurves.npy', initial_loadings_interpolateCurves)
    np.save(f'{initialResultPath}/common/reverse_initial_loadings_trueCurves.npy', reverse_initial_loadings_trueCurves)
    np.save(f'{initialResultPath}/common/reverse_initial_loadings_processCurves.npy', reverse_initial_loadings_processCurves)
    np.save(f'{initialResultPath}/common/reverse_initial_loadings_interpolateCurves.npy', reverse_initial_loadings_interpolateCurves)      
    
    # np.save(f'{iterationResultPath}/common/iteration_loadings_trueCurves.npy', iteration_loadings_trueCurves)
    # np.save(f'{iterationResultPath}/common/iteration_loadings_processCurves.npy', iteration_loadings_processCurves)
    # np.save(f'{iterationResultPath}/common/iteration_loadings_interpolateCurves.npy', iteration_loadings_interpolateCurves)
    # np.save(f'{iterationResultPath}/common/reverse_iteration_loadings_trueCurves.npy', reverse_iteration_loadings_trueCurves)
    # np.save(f'{iterationResultPath}/common/reverse_iteration_loadings_processCurves.npy', reverse_iteration_loadings_processCurves)
    # np.save(f'{iterationResultPath}/common/reverse_iteration_loadings_interpolateCurves.npy', reverse_iteration_loadings_interpolateCurves)

    prepared_data = {
        'initial_length': initial_length,
        'iteration_length': iteration_length,
        'exp_curve': exp_curve,
        'initialResultPath': initialResultPath,
        'iterationResultPath': iterationResultPath,
        'stage_CurvesList': stage_CurvesList,
        'stacked_exp_stress': stacked_exp_stress,
        'initial_loadings_trueCurves': initial_loadings_trueCurves, 
        'initial_loadings_processCurves': initial_loadings_processCurves,
        'initial_loadings_interpolateCurves': initial_loadings_interpolateCurves,
        'reverse_initial_loadings_trueCurves': reverse_initial_loadings_trueCurves,
        'reverse_initial_loadings_processCurves': reverse_initial_loadings_processCurves,
        'reverse_initial_loadings_interpolateCurves': reverse_initial_loadings_interpolateCurves,        
        'iteration_loadings_trueCurves': iteration_loadings_trueCurves, 
        'iteration_loadings_processCurves': iteration_loadings_processCurves,
        'iteration_loadings_interpolateCurves': iteration_loadings_interpolateCurves,
        'reverse_iteration_loadings_trueCurves': reverse_iteration_loadings_trueCurves,
        'reverse_iteration_loadings_processCurves': reverse_iteration_loadings_processCurves,
        'reverse_iteration_loadings_interpolateCurves': reverse_iteration_loadings_interpolateCurves,
        'combined_loadings_trueCurves': combined_loadings_trueCurves,
        'combined_loadings_processCurves': combined_loadings_processCurves,
        'combined_loadings_interpolateCurves': combined_loadings_interpolateCurves,
        'reverse_combined_loadings_trueCurves': reverse_combined_loadings_trueCurves,
        'reverse_combined_loadings_processCurves': reverse_combined_loadings_processCurves,
        'reverse_combined_loadings_interpolateCurves': reverse_combined_loadings_interpolateCurves,
    }
    #time.sleep(180)
    return prepared_data

if __name__ == '__main__':
    info = stage0_config.main()
    main_prepareData(info)
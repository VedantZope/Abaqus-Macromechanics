# External libraries
import os
import numpy as np
import stage0_config as stage0_config 
import stage2_prepare_data
import stage3_trainRegressor_withoutOptimizer
import stage4_stages_analysis
from modules.SIM_damask2 import *
from stage2_prepare_data import * 
from modules.preprocessing import *
from modules.stoploss import *
from modules.helper import *
from optimizers.GA import *
from optimizers.ANN import *
from prettytable import PrettyTable
from sklearn.preprocessing import StandardScaler
from retrainingRegressor import *

# Three optimization stages for-loop

def main_withoutOptimizer(info, prepared_data, stages_data, trained_models):

    server = info['server']
    loadings = info['loadings']
    CPLaw = info['CPLaw']
    convertUnit = info['convertUnit']
    initialSims = info['initialSims']
    curveIndex = info['curveIndex']
    projectPath = info['projectPath']
    optimizeStrategy = info['optimizeStrategy']
    optimizerName = info['optimizerName']
    param_info = info['param_info']
    param_info_GA = info['param_info_GA']
    param_info_BO = info['param_info_BO']
    param_info_BOSS = info['param_info_BOSS']
    param_info_PSO = info['param_info_PSO']
    param_info_filtered = info['param_info_filtered']
    param_info_searching =  info['param_info_searching']
    logPath = info['logPath']
    material = info['material']
    method = info['method']
    exampleLoading = info['exampleLoading']
    searchingSpace = info['searchingSpace']
    roundContinuousDecimals = info['roundContinuousDecimals']
    linearYieldingDev = info['linearYieldingDev']
    linearHardeningDev = info['linearHardeningDev'] 
    nonlinearHardeningDev = info['nonlinearHardeningDev']
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
    trainingParams = info['trainingParams']
    optimize_loadings = info['optimize_loadings']

    iteration_length = prepared_data['iteration_length']
    exp_curve = prepared_data['exp_curve']
    initialResultPath = prepared_data['initialResultPath']
    iterationResultPath = prepared_data['iterationResultPath']
    stage_CurvesList = prepared_data['stage_CurvesList']

    iteration_loadings_trueCurves = prepared_data['iteration_loadings_trueCurves']
    iteration_loadings_processCurves = prepared_data['iteration_loadings_processCurves']
    iteration_loadings_interpolateCurves = prepared_data['iteration_loadings_interpolateCurves']
    reverse_iteration_loadings_trueCurves = prepared_data['reverse_iteration_loadings_trueCurves']
    reverse_iteration_loadings_processCurves = prepared_data['reverse_iteration_loadings_processCurves']
    reverse_iteration_loadings_interpolateCurves = prepared_data['reverse_iteration_loadings_interpolateCurves']
    combined_loadings_trueCurves = prepared_data['combined_loadings_trueCurves']
    combined_loadings_processCurves = prepared_data['combined_loadings_processCurves']
    combined_loadings_interpolateCurves = prepared_data['combined_loadings_interpolateCurves']
    reverse_combined_loadings_trueCurves = prepared_data['reverse_combined_loadings_trueCurves']
    reverse_combined_loadings_processCurves = prepared_data['reverse_combined_loadings_processCurves']
    reverse_combined_loadings_interpolateCurves = prepared_data['reverse_combined_loadings_interpolateCurves']

    regressors = trained_models["regressors"]
    scalers = trained_models["scalers"]
    default_curve = trained_models['default_curve']

    deviationPercent_stages = stages_data['deviationPercent_stages']
    stopFunction_stages = stages_data['stopFunction_stages']
    SOO_lossFunction_stages = stages_data['SOO_lossFunction_stages']
    optimizeParams_stages = stages_data['optimizeParams_stages']
    weightsLoadings_stages = stages_data['weightsLoadings_stages']
    weightsConstitutive_stages = stages_data['weightsConstitutive_stages']
    parameterType_stages = stages_data['parameterType_stages']
    optimizeType_stages = stages_data['optimizeType_stages']
    ordinalUpper_stages = stages_data['ordinalUpper_stages']
    ordinalLower_stages = stages_data['ordinalLower_stages']
    ordinalNumber_stages = stages_data['ordinalNumber_stages']

    sim = SIM(info)
    sim.fileIndex = 0 # The file index is just 0. 
    # After an iteration, the code will delete the simulation folder on CSC and create a new 0 folder
    # This aims to prevent usage over disk memory so the optimizatio process can run without interruption  

    iterationIndex = iteration_length + 1
    
    info_optimize = copy.deepcopy(info)
    info_optimize['loadings'] = optimize_loadings

    #time.sleep(30)
    
    # stage_curve will save the result of the optimization in each iteration
    # stage_curve is appended to stage_curveList after each iteration
    # originally, stage_curve is the default_curve

    stage_curve = copy.deepcopy(default_curve)
    
    ################################
    # The three stage optimization #
    ################################

    contains_nonlinear = any(loading for loading in optimize_loadings if not loading.startswith("linear"))
    if contains_nonlinear:
        upperStage = 3
    else:
        upperStage = 2

    for stageNumber in range(0, upperStage):

        deviationPercent = deviationPercent_stages[stageNumber]
        stopFunction = stopFunction_stages[stageNumber]
        lossFunction = SOO_lossFunction_stages[stageNumber]
        optimizeParams = optimizeParams_stages[stageNumber]
        weightsLoadings = weightsLoadings_stages[stageNumber]
        weightsConstitutive = weightsConstitutive_stages[stageNumber]
        parameterType = parameterType_stages[stageNumber]
        optimizeType = optimizeType_stages[stageNumber]
        ordinalUpper = ordinalUpper_stages[stageNumber]
        ordinalLower = ordinalLower_stages[stageNumber]
        ordinalNumber = ordinalNumber_stages[stageNumber]

        printLog("\n" + 70 * "*" + "\n\n", logPath)
        printLog(f"{ordinalUpper} optimization stage: Optimize the {parameterType} parameters for the curve {CPLaw}{curveIndex}\n\n", logPath)
        
        #######################################################
        # The case when there are no parameters in this stage #
        #######################################################

        if len(optimizeParams) == 0:
            printLog(f"#### Stage {ordinalNumber} ####\n", logPath)
            printLog(f"There are no {parameterType} parameters to optimize\n", logPath)
            printLog(f"Saving {ordinalLower} stage result curves as the default curves\n", logPath)
            printLog(f"{ordinalUpper} optimization stage finished\n\n", logPath)
            
            # Saving the result of the ith optimization stage as the current curves
            np.save(f"{iterationResultPath}/common/stage{ordinalNumber}_curve.npy", stage_curve)

            printLog("The current result parameters\n", logPath)
            printDictParametersClean(stage_curve['parameters_dict'], param_info, paramsUnit, CPLaw, logPath)
        
        ########################################################
        # The case when this stage has already finished before #
        ########################################################
        
        elif os.path.exists(f"{iterationResultPath}/common/stage{ordinalNumber}_curve.npy"):
            printLog(f"#### Stage {ordinalNumber} ####\n", logPath)
            printLog(f"The file stage{ordinalNumber}_curve.npy detected, which means the {ordinalLower} stage has finished\n", logPath)
            printLog(f"{ordinalUpper} optimization stage finished\n\n", logPath)
            
            # Loading the existing curves
            stage_curve = np.load(f"{iterationResultPath}/common/stage{ordinalNumber}_curve.npy", allow_pickle=True).tolist()
            
            printLog("The current result parameters\n", logPath)
            printDictParametersClean(stage_curve['parameters_dict'], param_info, paramsUnit, CPLaw, logPath)
        
        #####################################################################
        # If not either cases, then calibrate the parameters for this stage #
        #####################################################################
        
        else: 
            printLog(f"#### Stage {ordinalNumber} ####\n", logPath)
            printLog(f"Optimizing the parameters {', '.join(optimizeParams)}\n", logPath)
            printLog(f"{ordinalUpper} optimization stage starts\n\n", logPath)
            
            # Calculate whether the default curves satisfies all loadings
            allLoadingsSatisfied, notSatisfiedLoadings = stopFunction(exp_curve["interpolate"], stage_curve["interpolate"], optimize_loadings, deviationPercent)
            while not allLoadingsSatisfied: 
                # The loadings that do not satisfy the deviation percentage
                printLog(f"#### Stage {ordinalNumber} ####\n", logPath)
                printLog(f"The loadings not satisfied are", logPath)
                for loading in notSatisfiedLoadings:
                    if optimizeType == "linear yielding":
                        printLog(f"{loading}: Exp yield: {exp_curve['interpolate'][loading]['stress'][1]} MPa | Sim yield: {stage_curve['interpolate'][loading]['stress'][1]} MPa", logPath)
                    else:
                        printLog(loading, logPath)
                printLog("\n", logPath)
                ########################################################
                # Outer while loop for checking simulation convergence #
                ########################################################
                
                # Setting converging to False. 
                converging = False
                
                while not converging:
                    # Initialize the optimizer
                    default_params = stage_curve["parameters_dict"]
                    optimizer.initializeOptimizer(default_params, optimizeParams, lossFunction, weightsLoadings, weightsConstitutive)
                    start = time.time()
                    # Running the optimizer
                    optimizer.run()
                    end = time.time()
                    # Output the best params found by the optimizer 
                    # bestParams contains 3 keys: solution_dict, solution_tuple and solution_loss
                    bestParams = optimizer.outputResult()
                    
                    ######################################################
                    # Inner while loop for checking repeated parameters #
                    #####################################################
                    
                    # Check if these params is already in the results. If yes, make the optimizer runs again
                    
                    while bestParams['solution_tuple'] in reverse_combined_loadings_interpolateCurves.keys():
                        printLog(f"#### Stage {ordinalNumber} ####\n", logPath)
                        printLog(f"#### (Curve {CPLaw}{curveIndex}) Iteration {iterationIndex} ####\n", logPath)
                        printLog(f"Searching time by {optimizerName}: {round(end - start, 2)}s\n\n", logPath)
                        printLog(f"The best candidate parameters found by {optimizerName}\n", logPath)
                        printLog("The current result parameters\n", logPath)
                        
                        printDictCalibratedParametersClean(bestParams['solution_dict'], optimizeParams_stages, stageNumber, param_info, paramsUnit, CPLaw, logPath)

                        printLog(f"Parameters already probed. {optimizerName} needs to run again to obtain new parameters\n", logPath)
                        printLog(f"Retraining ANN with slightly different configurations to prevent repeated parameters\n", logPath)
                        printLog(f"The number of combined interpolate curves is {len(combined_loadings_interpolateCurves[exampleLoading])}\n\n", logPath)
                        
                        # In order to prevent repeated params, retraining ANN with slightly different configuration
                        retrainingRegressor(optimizeType, loadings, regressors, scalers, combined_loadings_interpolateCurves, info)
                        
                        # Initialize the optimizer again
                        optimizer.initializeOptimizer(default_params, optimizeParams, lossFunction, weightsLoadings, weightsConstitutive)
                        start = time.time()
                        # Running the optimizer
                        optimizer.run()
                        end = time.time()
                        # Output the best params found by the optimizer again
                        bestParams = optimizer.outputResult()
                    
                    #**************************************#
                    # Outside the inner while loop of repeated parameters
                    
                    printLog(f"#### Stage {ordinalNumber} ####\n", logPath)
                    printLog(f"#### (Curve {CPLaw}{curveIndex}) Iteration {iterationIndex} ####\n", logPath)
                    printLog(f"Searching time by {optimizerName}: {round(end - start, 2)}s\n\n", logPath)
                    printLog(f"The best candidate parameters found by {optimizerName}\n", logPath)
                    
                    printDictCalibratedParametersClean(bestParams['solution_dict'], optimizeParams_stages, stageNumber, param_info, paramsUnit, CPLaw, logPath)
                    
                    printLog(f"This is new parameters. Loss of the best candidate parameters: {bestParams['solution_loss']}\n\n", logPath)                   
                    printLog(f"#### Stage {ordinalNumber} ####\n", logPath)
                    printLog(f"Running iteration {iterationIndex} simulation\n\n", logPath)
                    
                    time.sleep(30)

                    # Running a single iteration simulation and extracting the iteration simulation result
                    converging, one_new_iteration_trueCurves, one_new_iteration_processCurves = sim.run_iteration_simulations(bestParams['solution_dict'])
                    
                    if not converging:
                        printLog(f"#### Stage {ordinalNumber} ####\n", logPath)
                        printLog(f"#### (Curve {CPLaw}{curveIndex}) Iteration {iterationIndex} ####\n", logPath)
                        printLog("Iteration simulation has not converged. Rerunning the optimizer to obtain another set of candidate parameters\n\n", logPath)
                        printLog(f"Retraining ANN with slightly different configurations to prevent nonconverging parameters\n", logPath)
                        printLog(f"The number of combined interpolate curves is {len(combined_loadings_interpolateCurves[exampleLoading])}\n\n", logPath)
                        # In order to prevent nonconverging params, retraining ANN with slightly different configuration
                        retrainingRegressor(optimizeType, loadings, regressors, scalers, combined_loadings_interpolateCurves, info)
                #**************************************#
                # Outside the while loop of nonconvergence
                printLog(f"#### Stage {ordinalNumber} ####\n", logPath)
                printLog(f"#### (Curve {CPLaw}{curveIndex}) Iteration {iterationIndex} ####\n", logPath)
                printLog("Iteration simulation has converged. Saving the one new iteration simulation curves\n\n", logPath)
                
                
                # Update the iteration curves 
                for loading in loadings:
                    iteration_loadings_trueCurves[loading].update(one_new_iteration_trueCurves[loading])
                    iteration_loadings_processCurves[loading].update(one_new_iteration_processCurves[loading])
                
                # Update the reverse iteration curves
                reverse_iteration_loadings_trueCurves.update(reverseAsParamsToLoading(one_new_iteration_trueCurves, loadings, exampleLoading))
                reverse_iteration_loadings_processCurves.update(reverseAsParamsToLoading(one_new_iteration_processCurves, loadings, exampleLoading))
                
                # Update the current candidate curves (stage_curve) and interpolate iteration curves
                

                stage_curve = copy.deepcopy(stage_curve)
                stage_curve["iteration"] = iterationIndex
                stage_curve["stage"] = stageNumber
                stage_curve["parameters_tuple"] = bestParams["solution_tuple"]
                stage_curve["parameters_dict"] = bestParams["solution_dict"]
                stage_curve["true"] = reverse_iteration_loadings_trueCurves[bestParams["solution_tuple"]]
                stage_curve["process"] = reverse_iteration_loadings_processCurves[bestParams["solution_tuple"]]
    
                for loading in loadings:
                    stage_curve["interpolate"][loading] = {
                        "strain": exp_curve["interpolate"][loading]["strain"], 
                        "stress": interpolatingStress(stage_curve["process"][loading]["strain"], stage_curve["process"][loading]["stress"], exp_curve["interpolate"][loading]["strain"], loading).reshape(-1)
                    }

                predicted_curve = {}
                predicted_curve['interpolate'] = {}
                for loading in loadings:
                    predictedParams = scalers[loading].transform(np.array(list(stage_curve["parameters_dict"].values())).reshape(1, -1))
                    #print(predictedParams)
                    predicted_curve['interpolate'][loading] = {}
                    predicted_curve['interpolate'][loading]['stress'] = regressors[loading].predictOneDimension(predictedParams).flatten()
                    #print(predicted_curve['interpolate'][loading]['stress'])
                    #time.sleep(60)

                stage_curve["true_yielding_loss"] = calculateYieldingLoss(exp_curve["interpolate"], stage_curve["interpolate"], loadings)
                stage_curve["true_hardening_loss"] = calculateHardeningLoss(exp_curve["interpolate"], stage_curve["interpolate"], loadings)
                stage_curve["predicted_yielding_loss"] = calculateYieldingLoss(exp_curve["interpolate"], predicted_curve["interpolate"], loadings)
                stage_curve["predicted_hardening_loss"] = calculateHardeningLoss(exp_curve["interpolate"], predicted_curve["interpolate"], loadings)
        
                # Adding the one new curve
                stage_CurvesList.append(stage_curve)
                np.save(f"{iterationResultPath}/common/stage_CurvesList.npy", stage_CurvesList)
                  
                # Update iteration_interpolateCurves
                for loading in loadings:
                    iteration_loadings_interpolateCurves[loading][stage_curve["parameters_tuple"]] = stage_curve["interpolate"][loading]
                
                # Update reverse_iteration_interpolateCurves
                reverse_iteration_loadings_interpolateCurves[stage_curve["parameters_tuple"]] = stage_curve["interpolate"]
                
                # Update combined_loadings_interpolateCurves
                for loading in loadings:
                    combined_loadings_interpolateCurves[loading][stage_curve["parameters_tuple"]] = stage_curve["interpolate"][loading]

                # Update reverse_combined_loadings_interpolateCurves
                reverse_combined_loadings_interpolateCurves[stage_curve["parameters_tuple"]] = stage_curve["interpolate"]

                
                # Saving the updated iteration curves
                for loading in loadings: 
                    np.save(f"{iterationResultPath}/{loading}/iteration_trueCurves.npy", iteration_loadings_trueCurves[loading])
                    np.save(f"{iterationResultPath}/{loading}/iteration_processCurves.npy", iteration_loadings_processCurves[loading])
                np.save(f"{iterationResultPath}/common/iteration_loadings_trueCurves.npy", iteration_loadings_trueCurves)
                np.save(f"{iterationResultPath}/common/iteration_loadings_processCurves.npy", iteration_loadings_processCurves)
                np.save(f"{iterationResultPath}/common/iteration_loadings_interpolateCurves.npy", iteration_loadings_interpolateCurves)
                np.save(f"{iterationResultPath}/common/reverse_iteration_loadings_trueCurves.npy", reverse_iteration_loadings_trueCurves)
                np.save(f"{iterationResultPath}/common/reverse_iteration_loadings_processCurves.npy", reverse_iteration_loadings_processCurves)
                np.save(f"{iterationResultPath}/common/reverse_iteration_loadings_interpolateCurves.npy", reverse_iteration_loadings_interpolateCurves)

                printLog(f"#### Stage {ordinalNumber} ####\n", logPath)
                printLog(f"#### (Curve {CPLaw}{curveIndex}) Iteration {iterationIndex} ####\n", logPath)
                printLog(f"The new iteration true and predicted loss", logPath)
                printLoss(stage_curve, optimizeType, loadings, logPath)

                printLog(f"#### Stage {ordinalNumber} ####\n", logPath)
                printLog(f"#### (Curve {CPLaw}{curveIndex}) Iteration {iterationIndex} ####\n", logPath)
                printLog("Starting to retrain the ANN for all loadings\n", logPath)
                printLog(f"The number of combined interpolate curves is {len(combined_loadings_interpolateCurves[exampleLoading])}\n\n", logPath)
           
                # Retraining all the ANN
                retrainingRegressor(optimizeType, loadings, regressors, scalers, combined_loadings_interpolateCurves, info)

                printLog(f"#### Stage {ordinalNumber} ####\n", logPath)
                printLog(f"#### (Curve {CPLaw}{curveIndex}) Iteration {iterationIndex} ####\n", logPath)
                printLog(f"Finish training ANN for all loadings\n\n", logPath)
                            
                # Calculate whether the default curves satisfies all loadings
                allLoadingsSatisfied, notSatisfiedLoadings = stopFunction(exp_curve["interpolate"], stage_curve["interpolate"], optimize_loadings, deviationPercent)
            
                iterationIndex += 1
            #**************************************#
            
            # Saving the succeeding iteration
            np.save(f"{iterationResultPath}/common/stage{ordinalNumber}_curve.npy", stage_curve)
            
            # Outside the while loop of allLoadingsSatisfied
            printLog(f"#### Stage {ordinalNumber} ####\n", logPath)
            printLog(f"#### (Curve {CPLaw}{curveIndex}) Iteration {iterationIndex} ####\n", logPath)
            printLog(f"All loadings have successfully satisfied the deviation percentage\n", logPath)
            printLog(f"Succeeded iteration: {iterationIndex}\n", logPath)

            printLog(f"The {ordinalLower} stage parameter solution is:\n", logPath)
            printDictParametersClean(stage_curve['parameters_dict'], param_info, paramsUnit, CPLaw, logPath)  
            printLog(f"{ordinalUpper} optimization stage finished\n\n", logPath)
                
if __name__ == '__main__':
    info = stage0_config.main()
    prepared_data = stage2_prepare_data.main_prepareData(info)
    trained_models = stage3_trainRegressor_withoutOptimizer.main_trainANN(info, prepared_data)
    stages_data = stage4_stages_analysis.main_stagesAnalysis(info, prepared_data)
    main_withoutOptimizer(info, prepared_data, stages_data, trained_models)
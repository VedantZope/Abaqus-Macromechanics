# External libraries
import os
import numpy as np
import stage0_config as stage0_config
import stage2_prepare_data
#import stages_analysis
#import optimization_stages
from modules.SIM_damask2 import *
from stage2_prepare_data import * 
from modules.preprocessing import *
from modules.stoploss import *
from modules.helper import *
from optimizers.GA import *
from optimizers.ANN import *
from optimizers.scaler import *
from optimizers.XGBOOST import * 
from prettytable import PrettyTable
from sklearn.preprocessing import StandardScaler

def main_trainRegressor_withOptimizer(info, prepared_data, logging):
    logPath = info['logPath']
    loadings = info['loadings']
    CPLaw = info['CPLaw']
    convertUnit = info['convertUnit']
    curveIndex = info['curveIndex']
    param_info = info['param_info']
    param_info_filtered = info['param_info_filtered']
    paramsUnit = info['paramsUnit']
    material = info['material']
    loadings = info['loadings']
    trainingParams = info['trainingParams']
    using_initial_params = info['using_initial_params']
    optimize_loadings = info['optimize_loadings']
    exampleLoading = info['exampleLoading']
    weightsYieldingConstitutive = info['weightsYieldingConstitutive']
    weightsHardeningConstitutive = info['weightsHardeningConstitutive']
    weightsYieldingLinearLoadings = info['weightsYieldingLinearLoadings']
    weightsHardeningLinearLoadings = info['weightsHardeningLinearLoadings']
    weightsHardeningAllLoadings = info['weightsHardeningAllLoadings']
    regressorName = info['regressorName']
    verbose = info['verbose']
    optimizeHyperparams = info['optimizeHyperparams']
    projectPath = info['projectPath']

    initial_length = prepared_data['initial_length']
    combined_loadings_interpolateCurves = prepared_data['combined_loadings_interpolateCurves']
    exp_curve = prepared_data['exp_curve']

    iterationResultPath = prepared_data['iterationResultPath']

    initial_loadings_trueCurves = prepared_data['initial_loadings_trueCurves']
    initial_loadings_processCurves = prepared_data['initial_loadings_processCurves']
    initial_loadings_interpolateCurves = prepared_data['initial_loadings_interpolateCurves']
    reverse_initial_loadings_trueCurves = prepared_data['reverse_initial_loadings_trueCurves']
    reverse_initial_loadings_processCurves = prepared_data['reverse_initial_loadings_processCurves']
    reverse_initial_loadings_interpolateCurves = prepared_data['reverse_initial_loadings_interpolateCurves']
    
    if logging:
        printLog("\n" + 70 * "*" + "\n\n", logPath)
        printLog(f"Step 2: Train the regressors for all loadings with the initial simulations of curve {CPLaw}{curveIndex}\n", logPath)
        printLog(f"Regressor model: (parameters) -> (stress values at interpolating strain points)\n", logPath)    

    # The regressors for each loading condition
    regressors = {}

    featureMatrixScaling = np.zeros((2, len(list(param_info_filtered.keys()))))
    powerList = np.zeros(len(list(param_info_filtered.keys())))
    for index, parameter in enumerate(list(param_info_filtered.keys())):
        featureMatrixScaling[:, index] = np.array([param_info_filtered[parameter]["generalLow"], param_info_filtered[parameter]["generalHigh"]])
        powerList[index] = param_info_filtered[parameter]["power"]

    paramFeaturesOriginal = np.array([list(dict(params).values()) for params in list(combined_loadings_interpolateCurves[exampleLoading].keys())])
    scaler = CustomScaler(featureMatrixScaling, powerList)
    paramFeatures = scaler.transform(paramFeaturesOriginal)

    start = time.time()
    optimizedHyperparams = {}
    
    for loading in loadings:
        # All loadings share the same parameters, but different stress values
        paramFeatures = np.array([list(dict(params).values()) for params in list(combined_loadings_interpolateCurves[loading].keys())])

        stressLabels = np.array([strainstress["stress"] for strainstress in list(combined_loadings_interpolateCurves[loading].values())])
        #print(paramFeatures[0])
        #print(stressLabels[0])
        #time.sleep(30)
        # transforming the data

        paramFeatures = scaler.transform(paramFeatures)
        
        np.save(f"trained_models/{material}/{CPLaw}/paramFeatures_{loading}.npy", paramFeatures)
        np.save(f"trained_models/{material}/{CPLaw}/stressLabels_{loading}.npy", stressLabels)
        # Input and output size of the ANN
        inputSize = paramFeatures.shape[1]
        outputSize = stressLabels.shape[1]
        
        if regressorName == "ANN":
            savingPath = f"trained_models/{material}/{CPLaw}/model_{loading}.pth"
            trainingParams['savingPath'] = savingPath
            regressors[loading] = NeuralNetwork(inputSize, outputSize, trainingParams).to(device)
            regressors[loading].trainRegressor(paramFeatures, stressLabels, trainingParams, True)
            
            # Load the state with the lowest validation error
            checkpoint = torch.load(savingPath)
            regressors[loading].load_state_dict(checkpoint['model_state_dict'])
            best_train_loss = checkpoint['train_loss']
            best_val_loss = checkpoint['val_loss']
            
            stressLabels_pred = regressors[loading].predict(paramFeatures)
            error = regressors[loading].MSE_loss(stressLabels_pred, stressLabels)
                        
            if logging:
                printLog(f"\n------------ {loading} ------------\n", logPath)
                printLog(f"paramFeatures shape is {paramFeatures.shape}\n", logPath)
                printLog(f"stressLabels shape is {stressLabels.shape}\n", logPath)
                printLog(f"Training MSE loss: {best_train_loss}\n", logPath)
                printLog(f"Testing MSE loss: {best_val_loss}\n", logPath)
                printLog(f"Total MSE loss: {error}\n", logPath)
        
        elif regressorName == "XGBOOST":
            test_ratio = trainingParams[loading]['test_ratio']
            split_index = int(len(paramFeatures) * (1 - test_ratio))

            train_features = paramFeatures[:split_index]
            train_labels = stressLabels[:split_index]
            test_features = paramFeatures[split_index:]
            test_labels = stressLabels[split_index:]

            eval_set = [(test_features, test_labels)]
            regressors[loading] = XGBOOST(trainingParams[loading])
            optimizedHyperparams[loading] = regressors[loading].trainRegressor(train_features, train_labels, eval_set=eval_set, optimizeHyperparams=optimizeHyperparams, verbose=verbose)

            stressLabels_pred = regressors[loading].predict(paramFeatures)

            errorTrain = regressors[loading].MSE_loss(stressLabels_pred[:split_index], stressLabels[:split_index]) 
            errorTest = regressors[loading].MSE_loss(stressLabels_pred[split_index:], stressLabels[split_index:])
            error = regressors[loading].MSE_loss(stressLabels_pred, stressLabels)
            if logging:
                printLog(f"\n------------ {loading} ------------\n", logPath)
                printLog(f"paramFeatures shape is {paramFeatures.shape}\n", logPath)
                printLog(f"stressLabels shape is {stressLabels.shape}\n", logPath)
                printLog(f"MSE train error: {errorTrain}\n", logPath)
                printLog(f"MSE test error: {errorTest}\n", logPath)
                printLog(f"Total MSE error: {error}\n", logPath)
            #time.sleep(5)

    end = time.time()

    if logging:
        printLog(f"The number of combined interpolate curves is {len(combined_loadings_interpolateCurves[loading])}\n", logPath)
        printLog(f"Finish training ANN for all loadings of curve {CPLaw}{curveIndex}\n", logPath)
        printLog(f"Total training time: {round(end - start, 2)}s\n\n", logPath)

    if not os.path.exists(f"{iterationResultPath}/common/default_curve.npy"):
        tupleParamsStresses = list(reverse_initial_loadings_interpolateCurves.items())
        #sortedClosestYielding = list(sorted(tupleParamsStresses, key = lambda paramsStresses: lossYieldingAllLinear(exp_curve["interpolate"], paramsStresses[1], loadings, weightsYieldingLinearLoadings, weightsYieldingConstitutive)))
        sortedClosestHardening = list(sorted(tupleParamsStresses, key = lambda paramsStresses: lossHardeningAllLoadings_SOO(exp_curve["interpolate"], paramsStresses[1], loadings, weightsHardeningAllLoadings, weightsHardeningConstitutive)))

        # Obtaining the default hardening parameters
        parameters_tuple = sortedClosestHardening[0][0]
        parameters_dict = dict(parameters_tuple)
        default_curve = {}
        default_curve["iteration"] = 0
        default_curve["stage"] = 0
        default_curve["parameters_tuple"] = parameters_tuple
        default_curve["parameters_dict"] = parameters_dict
        default_curve["true"] = reverse_initial_loadings_trueCurves[parameters_tuple]
        default_curve["process"] = reverse_initial_loadings_processCurves[parameters_tuple]
        default_curve["interpolate"] = reverse_initial_loadings_interpolateCurves[parameters_tuple]
        default_curve["true_yielding_loss"] = calculateYieldingLoss(exp_curve["interpolate"], default_curve["interpolate"], loadings)
        default_curve["true_hardening_loss"] = calculateHardeningLoss(exp_curve["interpolate"], default_curve["interpolate"], loadings)
        predicted_curve = {}
        predicted_curve['interpolate'] = {}
        for loading in loadings:
            predictedParams = scaler.transform(np.array(list(default_curve["parameters_dict"].values())).reshape(1, -1))
            #print(predictedParams)
            predicted_curve['interpolate'][loading] = {}
            predicted_curve['interpolate'][loading]['stress'] = regressors[loading].predictOneDimension(predictedParams).flatten()
            #print(predicted_curve['interpolate'][loading]['stress'])
            #time.sleep(60)
        
        default_curve["predicted_yielding_loss"] = calculateYieldingLoss(exp_curve["interpolate"], predicted_curve["interpolate"], loadings)
        default_curve["predicted_hardening_loss"] = calculateHardeningLoss(exp_curve["interpolate"], predicted_curve["interpolate"], loadings)
        # print(default_curve["true_yielding_loss"] )
        # print(default_curve["predicted_yielding_loss"] )
        # print(default_curve["true_hardening_loss"] )
        # print(default_curve["predicted_hardening_loss"] )
        np.save(f"{iterationResultPath}/common/default_curve.npy", default_curve)
    else:
        default_curve = np.load(f"{iterationResultPath}/common/default_curve.npy", allow_pickle=True).tolist()
        printLog("The file default_curve.npy exists. Loading the default curves\n", logPath)

    printLog(f"The default parameters for the optimization of curve {CPLaw}{curveIndex}\n", logPath)
    #print(default_curve["parameters_tuple"])
    printTupleParametersClean(default_curve["parameters_tuple"], param_info, paramsUnit, CPLaw, logPath)
    #time.sleep(60)
    
    # Using initial values from the config for params not targeted for calibration (no in search target column)
    if using_initial_params == "yes":
        for param in param_info_filtered:
            default_curve["parameters_dict"][param] = param_info_filtered[param]['initial']
        default_curve["parameters_tuple"] = tuple(default_curve["parameters_dict"].items())

    trained_models = {
        "regressors": regressors,
        "scaler": scaler,
        'default_curve': default_curve,
    }
    
    #print("Hello")
    #time.sleep(180)

    return trained_models

if __name__ == '__main__':
    info = stage0_config.main_config()
    prepared_data = stage2_prepare_data.main_prepareData(info)
    main_trainRegressor_withOptimizer(info, prepared_data, logging=True)
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
from optimizers.LSTM import *
from optimizers.scaler import * 
from prettytable import PrettyTable
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def main_trainLSTM_withoutOptimizer(info, prepared_data, logging):
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
    trainingParams_LSTM = info['trainingParams_LSTM']
    optimize_loadings = info['optimize_loadings']
    exampleLoading = info['exampleLoading']

    weightsYieldingConstitutive = info['weightsYieldingConstitutive']
    weightsHardeningConstitutive = info['weightsHardeningConstitutive']
    weightsYieldingLinearLoadings = info['weightsYieldingLinearLoadings']
    weightsHardeningLinearLoadings = info['weightsHardeningLinearLoadings']
    weightsHardeningAllLoadings = info['weightsHardeningAllLoadings']

    initial_length = prepared_data['initial_length']
    combined_loadings_interpolateCurves = prepared_data['combined_loadings_interpolateCurves']
    exp_curve = prepared_data['exp_curve']
    stacked_exp_stress = prepared_data['stacked_exp_stress']
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
        printLog(f"LSTM model: (stress values at interpolating strain points) -> (parameters)\n", logPath)
        
        numberOfHiddenLayers = trainingParams_LSTM['numberOfHiddenLayers']
        hiddenSize = trainingParams_LSTM['hiddenSize']
        
        dropoutRate = trainingParams_LSTM['dropoutRate']
        learning_rate = trainingParams_LSTM['learning_rate']
        weight_decay = trainingParams_LSTM['weight_decay']
        step_size = trainingParams_LSTM['step_size']
        gamma = trainingParams_LSTM['gamma']
        tolerance = trainingParams_LSTM['tolerance']
        patience = trainingParams_LSTM['patience']
        test_ratio = trainingParams_LSTM['test_ratio']
        batch_size = trainingParams_LSTM['batch_size']
        max_epochs = trainingParams_LSTM['max_epochs']

        stringMessage = "LSTM configuration:\n"

        logTable = PrettyTable()
        logTable.field_names = ["LSTM configurations", "Choice"]

        logTable.add_row(["Number of hidden layers", numberOfHiddenLayers])
        logTable.add_row(["Hidden layer size", hiddenSize])
        logTable.add_row(["Dropout rate", dropoutRate])
        logTable.add_row(["Learning rate", learning_rate])
        logTable.add_row(["weight_decay", weight_decay])
        logTable.add_row(["step_size", step_size])
        logTable.add_row(["gamma", gamma])
        logTable.add_row(["tolerance", tolerance])
        logTable.add_row(["patience", patience])
        logTable.add_row(["test_ratio", test_ratio])
        logTable.add_row(["batch_size", batch_size])
        logTable.add_row(["max_epochs", max_epochs])

        stringMessage += logTable.get_string()
        stringMessage += "\n"

        printLog(stringMessage, logPath)

    # The LSTM regressors for each loading condition
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
    
    stressList = []
    for loading in loadings:
        # All loadings share the same parameters, but different stress values
        stressLoadings = np.array([strainstress["stress"] for strainstress in list(combined_loadings_interpolateCurves[loading].values())])
        # print(stressLoadings.shape)
        stressList.append(stressLoadings)
        #print(stressLabels[0])
        #time.sleep(30)
        # transforming the data
    
    
    stressLabels = np.concatenate(stressList, axis=1)
    # print(stressLabels[1])
    #print(stressLabels.shape)
    #time.sleep(30)


    # Input and output size of the LSTM
    inputSize = stressLabels.shape[1]
    outputSize = paramFeatures.shape[1]
    # print(inputSize, outputSize) # 159, 10
        
    regressors[loading] = LSTM(inputSize, outputSize, trainingParams_LSTM).to(device)
    regressors[loading].trainLSTM(stressLabels, paramFeatures, trainingParams_LSTM, True)
    paramFeatures_pred = regressors[loading].predict(stressLabels)
    error = regressors[loading].MSE_loss(paramFeatures_pred, paramFeatures)
    
    #time.sleep(30)

    if logging:
        #printLog(f"------------ {loading} ------------\n", logPath)
        printLog(f"paramFeatures shape is {paramFeatures.shape}\n", logPath)
        printLog(f"stressLabels shape is {stressLabels.shape}\n", logPath)
        printLog(f"MSE error: {error}\n", logPath)

    end = time.time()

    if logging:
        printLog(f"The number of combined interpolate curves is {len(combined_loadings_interpolateCurves[loading])}\n", logPath)
        printLog(f"Finish training LSTM for all loadings of curve {CPLaw}{curveIndex}\n", logPath)
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
        default_curve["true_parameters_tuple"] = parameters_tuple
        default_curve["true_parameters_dict"] = parameters_dict
        default_curve["true"] = reverse_initial_loadings_trueCurves[parameters_tuple]
        default_curve["process"] = reverse_initial_loadings_processCurves[parameters_tuple]
        default_curve["interpolate"] = reverse_initial_loadings_interpolateCurves[parameters_tuple]
        default_curve["true_yielding_loss"] = calculateYieldingLoss(exp_curve["interpolate"], default_curve["interpolate"], loadings)
        default_curve["true_hardening_loss"] = calculateHardeningLoss(exp_curve["interpolate"], default_curve["interpolate"], loadings)

        stacked_sim_stress = np.hstack(list(map(lambda loading: exp_curve["interpolate"][loading]["stress"], loadings)))
        scaled_predicted_parameters = regressors[loading].predictOneDimension(stacked_sim_stress).flatten()
        predicted_parameters = scaler.inverse_transform(scaled_predicted_parameters.reshape(1, -1)).flatten()

        param_names = list(default_curve["true_parameters_dict"].keys())
        true_parameters = np.array(list(default_curve["true_parameters_dict"].values()))

        predicted_parameters_dict = dict(zip(param_names, predicted_parameters))
        predicted_parameters_tuple = tuple(zip(param_names, predicted_parameters))
        
        #print(default_curve['predicted_parameters'])
        default_curve["predicted_parameters_tuple"] = predicted_parameters_dict
        default_curve["predicted_parameters_dict"] = predicted_parameters_tuple

        scaled_true_parameters = scaler.transform(true_parameters.reshape(1, -1)).flatten()
        default_curve["predicted_parameters_loss"] = mean_squared_error(scaled_true_parameters, scaled_predicted_parameters)
        print(default_curve["predicted_parameters_tuple"])
        print(default_curve["predicted_parameters_dict"])
        print(default_curve["predicted_parameters_loss"])
        #print(predicted_curve['interpolate'][loading]['stress'])
        time.sleep(180)
        
        np.save(f"{iterationResultPath}/common/default_curve.npy", default_curve)
    else:
        default_curve = np.load(f"{iterationResultPath}/common/default_curve.npy", allow_pickle=True).tolist()
        printLog("The file default_curve.npy exists. Loading the default curves\n", logPath)

    printLog(f"The default parameters for the optimization of curve {CPLaw}{curveIndex}\n", logPath)
    printTupleParametersClean(default_curve["parameters_tuple"], param_info, paramsUnit, CPLaw, logPath)
    #time.sleep(60)
    
    # Comment these three lines to optimize using default params from existing curve 
    # instead of initial params from param_info.xlsx
    for param in param_info_filtered:
        default_curve["parameters_dict"][param] = param_info_filtered[param]['initial']
    default_curve["parameters_tuple"] = tuple(default_curve["parameters_dict"].items())

    trained_models = {
        "regressors": regressors,
        "scaler": scaler,
        'default_curve': default_curve,
    }

    time.sleep(180)

    return trained_models

if __name__ == '__main__':
    info = stage0_config.main_config()
    prepared_data = stage2_prepare_data.main_prepareData(info)
    main_trainLSTM_withoutOptimizer(info, prepared_data, logging=True)
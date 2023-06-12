# External libraries
import os
import numpy as np
import stage0_config as stage0_config
import stage2_prepare_data
from modules.SIM_damask2 import *
from modules.preprocessing import *
from modules.helper import *
from prettytable import PrettyTable

def main_stagesAnalysis(info, prepared_data):    
    server = info['server']
    loadings = info['loadings']
    CPLaw = info['CPLaw']
    convertUnit = info['convertUnit']
    initialSims = info['initialSims']
    curveIndex = info['curveIndex']
    projectPath = info['projectPath']
    optimizerName = info['optimizerName']
    param_info = info['param_info']
    param_info_filtered = info['param_info_filtered']
    param_info_searching =  info['param_info_searching']
    logPath = info['logPath']
    material = info['material']
    method = info['method']
    searchingSpace = info['searchingSpace']
    roundContinuousDecimals = info['roundContinuousDecimals']
    linearYieldingDev = info['linearYieldingDev']
    linearHardeningDev = info['linearHardeningDev'] 
    nonlinearHardeningDev = info['nonlinearHardeningDev']
    loadings = info['loadings']
    exampleLoading = info['exampleLoading']
    yieldingPoints = info['yieldingPoints']
    optimizeStrategy = info['optimizeStrategy']
    weightsYieldingConstitutive = info['weightsYieldingConstitutive']
    weightsHardeningConstitutive = info['weightsHardeningConstitutive']
    optimize_loadings = info['optimize_loadings']

    weightsYieldingLinearLoadings = info['weightsYieldingLinearLoadings']
    weightsHardeningLinearLoadings = info['weightsHardeningLinearLoadings']
    weightsHardeningAllLoadings = info['weightsHardeningAllLoadings']

    ranksYieldingLinearLoadings = info['ranksYieldingLinearLoadings']
    ranksHardeningLinearLoadings = info['ranksHardeningLinearLoadings']
    ranksHardeningAllLoadings = info['ranksHardeningAllLoadings']

    iteration_length = prepared_data['iteration_length']

    printLog("\n" + 70 * "*" + "\n\n", logPath)
    printLog(f"Step 3: Assessment of the optimization stages of curve {CPLaw}{curveIndex}\n", logPath)

    # allParams = list(param_info_filtered.keys())
    allParams = list(param_info_searching.keys())

    yieldingParams = list(filter(lambda param: param_info[param]["type"] == "yielding", allParams))
    linearHardeningParams = list(filter(lambda param: param_info[param]["type"] == "linear_hardening", allParams))
    nonlinearHardeningParams = list(filter(lambda param: param_info[param]["type"] == "nonlinear_hardening", allParams))
    
    printLog(f"The yielding parameters are {yieldingParams}\n", logPath)
    printLog(f"The linear hardening parameters are {linearHardeningParams}\n", logPath)
    printLog(f"The nonlinear hardening parameters are {nonlinearHardeningParams}\n\n", logPath)    
    
    if len(yieldingParams) == 0:
        printLog("There are yielding parameters\n", logPath)
        printLog("1st stage optimization not required\n", logPath)
    else:
        printLog(f"There are {len(yieldingParams)} yielding parameters\n", logPath)
        printLog("1st stage optimization required\n", logPath)
    
    if len(linearHardeningParams) == 0:
        printLog("There are no linear hardening parameters\n", logPath)
        printLog("2nd stage optimization not required\n", logPath)
    else:
        printLog(f"There are {len(linearHardeningParams)} linear hardening parameters\n", logPath)
        printLog("2nd stage optimization required\n", logPath)

    if len(nonlinearHardeningParams) == 0:
        printLog("There are no nonlinear hardening parameters\n", logPath)
        printLog("3rd stage optimization not required\n\n", logPath)
    else:
        printLog(f"There are {len(nonlinearHardeningParams)} small hardening parameters\n", logPath)
        printLog("3rd stage optimization required\n\n", logPath)

    # ----------------------------------------------------------------------------
    #   Three optimization stage: Optimize the parameters for the curves in parallel 
    # ----------------------------------------------------------------------------

    #print(optimize_loadings)
    # filtered out linear loadings from optimize_loadings
    optimize_linear_loadings = [loading for loading in optimize_loadings if loading.startswith("linear")]
    optimize_nonlinear_loadings = [loading for loading in optimize_loadings if loading.startswith("nonlinear")]
    
    assert len(optimize_linear_loadings) >= 1, "There should be at least one linear loading"
    if len(optimize_nonlinear_loadings) == 0:
        printLog("There are no nonlinear loadings\n", logPath)
        printLog("3rd stage optimization not required\n\n", logPath)
    else:
        printLog(f"There are {len(optimize_nonlinear_loadings)} nonlinear loadings\n", logPath)
        printLog("3rd stage optimization required\n\n", logPath)

    
    optimizeLoadings_stages = [optimize_linear_loadings, optimize_linear_loadings, optimize_loadings]
    
    deviationPercent_stages = [linearYieldingDev, linearHardeningDev, nonlinearHardeningDev]
    stopFunction_stages = [insideYieldingDevAllLinear, insideHardeningDevAllLinear, insideHardeningDevAllLoadings]
    optimizeParams_stages = [yieldingParams, linearHardeningParams, nonlinearHardeningParams]
    weightsConstitutive_stages = [weightsYieldingConstitutive, weightsHardeningConstitutive, weightsHardeningConstitutive]
    parameterType_stages = ["linear yielding", "linear hardening", "nonlinear hardening"]
    optimizeType_stages = ["linear yielding", "linear hardening", "nonlinear hardening"]
    ordinalUpper_stages = ["First", "Second", "Third"]
    ordinalLower_stages = ["first", "second", "third"]
    ordinalNumber_stages = ["1","2","3"]
    if optimizeStrategy == "SOO_withOptimizer":
        lossFunction_stages = [lossYieldingAllLinear_SOO, lossHardeningAllLinear_SOO, lossHardeningAllLoadings_SOO]
    elif optimizeStrategy == "MOO_withOptimizer":
        lossFunction_stages = [lossYieldingOneLoading_MOO, lossHardeningOneLoading_MOO, lossHardeningOneLoading_MOO]
    weightsLoadings_stages = [weightsYieldingLinearLoadings, weightsHardeningLinearLoadings, weightsHardeningAllLoadings]
    ranksLoadings_stages = [ranksYieldingLinearLoadings, ranksHardeningLinearLoadings, ranksHardeningAllLoadings]


    stages_data = {
        'deviationPercent_stages':deviationPercent_stages,
        'optimizeLoadings_stages':optimizeLoadings_stages,
        'stopFunction_stages': stopFunction_stages,
        'lossFunction_stages': lossFunction_stages,
        'optimizeParams_stages':optimizeParams_stages,
        'weightsConstitutive_stages': weightsConstitutive_stages,
        'weightsLoadings_stages': weightsLoadings_stages,
        'ranksLoadings_stages': ranksLoadings_stages,
        'parameterType_stages':parameterType_stages,
        'optimizeType_stages':optimizeType_stages,
        'ordinalUpper_stages':ordinalUpper_stages,
        'ordinalLower_stages':ordinalLower_stages,
        'ordinalNumber_stages':ordinalNumber_stages,
    }

    #time.sleep(180)

    return stages_data

if __name__ == '__main__':
    info = stage0_config.main()
    prepared_data = stage2_prepare_data.main_prepareData(info)
    stages_data = main_stagesAnalysis(info, prepared_data)
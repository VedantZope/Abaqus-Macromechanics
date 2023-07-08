import os
import time
import pandas as pd
import numpy as np
from time import sleep
from prettytable import PrettyTable
from stage0_initialize_directory import *
from modules.helper import *
from modules.hardeningLaws import *
import copy

############################################################
#                                                          #
#        ABAQUS HARDENING LAW PARAMETER CALIBRATION        #
#   Tools required: Abaqus and Finnish Supercomputer CSC   #
#                                                          #
############################################################

# ------------------------------------#
#   Stage 0: Recording configurations #
# ------------------------------------#

def main_config():

    #########################
    # Global configurations #
    #########################

    globalConfig = pd.read_excel("configs/global_config.xlsx", nrows=1, engine="openpyxl")
    globalConfig = globalConfig.T.to_dict()[0]
    optimizeStrategy = globalConfig["optimizeStrategy"]
    material = globalConfig["material"]
    optimizerName = globalConfig["optimizerName"]
    hardeningLaw = globalConfig["hardeningLaw"]
    deviationPercent = globalConfig["deviationPercent"]
    geometry = globalConfig["geometry"]
    curveIndex = globalConfig["curveIndex"]
    generateParams = globalConfig["generateParams"]
    numberOfInitialSims = globalConfig["numberOfInitialSims"]
    initialSimsSpacing = globalConfig["initialSimsSpacing"]
    SLURM_iteration = globalConfig["SLURM_iteration"]

    if optimizeStrategy == "SOO":
        (
            projectPath, 
            logPath, 
            paramInfoPath, 
            resultPath, 
            simPath, 
            templatePath, 
            targetPath
        ) = initialize_directory(optimizeStrategy, material, hardeningLaw, geometry, curveIndex)

    elif optimizeStrategy == "MOO":
        geometries = geometry.split(",")
        (
            projectPath, 
            logPath, 
            paramInfoPath, 
            resultPath, 
            simPath, 
            templatePath, 
            targetPath
        ) = initialize_directory(optimizeStrategy, material, hardeningLaw, geometries, curveIndex)
    
    #################################
    # Plastic strain configurations #
    #################################
    
    truePlasticStrainConfig = pd.read_excel(f"configs/truePlasticStrain_{hardeningLaw}_config.xlsx",engine="openpyxl")
    ranges_and_increments = []

    # Iterate over each row in the DataFrame
    for index, row in truePlasticStrainConfig.iterrows():
        # Append a tuple containing the strainStart, strainEnd, and strainStep to the list
        ranges_and_increments.append((row['strainStart'], row['strainEnd'], row['strainStep']))
        
    truePlasticStrain = np.array([])

    # Iterate through each range and increment
    for i, (start, end, step) in enumerate(ranges_and_increments):
        # Skip the start value for all ranges after the first one
        if i > 0:
            start += step
        # Create numpy array for range
        strain_range = np.arange(start, end + step, step)
        strain_range = np.around(strain_range, decimals=6)
        # Append strain_range to strain_array
        truePlasticStrain = np.concatenate((truePlasticStrain, strain_range))

    np.save(f"configs/truePlasticStrain_{hardeningLaw}.npy", truePlasticStrain)

    ##################################
    # Parameter bound configurations #
    ##################################

    paramConfig = pd.read_excel(f"{paramInfoPath}/paramInfo.xlsx", engine="openpyxl")
    paramConfig.set_index("parameter", inplace=True)
    paramConfig = paramConfig.T.to_dict()
    for param in paramConfig:
        paramConfig[param]['exponent'] = float(paramConfig[param]['exponent'])

    ###########################
    # Information declaration #
    ###########################

    info = {
        'projectPath': projectPath,
        'logPath': logPath,
        'paramInfoPath': paramInfoPath,
        'resultPath': resultPath,
        'simPath': simPath,
        'targetPath': targetPath,
        'templatePath': templatePath,
        'optimizeStrategy': optimizeStrategy,
        'numberOfInitialSims': numberOfInitialSims,
        'generateParams': generateParams,
        'initialSimsSpacing': initialSimsSpacing,
        'material': material,
        'hardeningLaw': hardeningLaw,
        'curveIndex': curveIndex,
        'optimizerName': optimizerName,
        'paramConfig': paramConfig,
        'deviationPercent': deviationPercent,
        'truePlasticStrain': truePlasticStrain,
        'SLURM_iteration': SLURM_iteration
    }

    if optimizeStrategy == "SOO":
        info['geometry'] = geometry
    if optimizeStrategy == "MOO":
        info['geometries'] = geometries

    ###############################################
    #  Printing the configurations to the console #
    ###############################################

    printLog(f"\nWelcome to the Abaqus parameter calibration project\n\n", logPath)
    printLog(f"The configurations you have chosen: \n", logPath)
    
    logTable = PrettyTable()

    logTable.field_names = ["Global Configs", "User choice"]
    logTable.add_row(["SLURM iteration", SLURM_iteration])
    logTable.add_row(["Number of initial sims", numberOfInitialSims])
    logTable.add_row(["Optimize strategy", optimizeStrategy])
    logTable.add_row(["Material", material])
    logTable.add_row(["Hardening law", hardeningLaw])
    logTable.add_row(["Curve index", curveIndex])
    if optimizeStrategy == "SOO":
        logTable.add_row(["Geometry", geometry])
    if optimizeStrategy == "MOO":
        geometryString = ",".join(geometries)
        logTable.add_row(["Geometries", geometryString])
    logTable.add_row(["Optimizer name", optimizerName])
    logTable.add_row(["Deviation percent", deviationPercent])

    printLog(logTable.get_string() + "\n", logPath)

    printLog("Generating necessary directories\n", logPath)
    printLog(f"The path to your main project folder is\n", logPath)
    printLog(f"{projectPath}\n", logPath)

    #############################
    # Returning the information #
    #############################

    #time.sleep(180)
    return info

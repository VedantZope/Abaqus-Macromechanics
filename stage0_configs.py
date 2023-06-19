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
    runInitialSims = globalConfig["runInitialSims"]
    numberOfInitialSims = globalConfig["numberOfInitialSims"]
    initialSimsSpacing = globalConfig["initialSimsSpacing"]

    projectPath, logPath, resultPath, simPath, templatePath, targetPath = initialize_directory(optimizeStrategy, material, geometry, hardeningLaw)
    
    ##################################
    # Parameter bound configurations #
    ##################################

    paramConfig = pd.read_excel(f"configs/{hardeningLaw}_paramInfo.xlsx", engine="openpyxl")
    paramConfig.set_index("parameter", inplace=True)
    paramConfig = paramConfig.T.to_dict()
    for param in paramConfig:
        paramConfig[param]['exponent'] = float(paramConfig[param]['exponent'])
        exponent = paramConfig[param]['exponent']
        paramConfig[param]['lowerBound'] = paramConfig[param]['lowerBound'] * exponent
        paramConfig[param]['upperBound'] = paramConfig[param]['upperBound'] * exponent
    
    #########################
    # Abaqus configurations #
    #########################
    abaqusConfig = pd.read_excel("abaqus_config.xlsx",engine="openpyxl")
    ranges_and_increments = []

    # Iterate over each row in the DataFrame
    for index, row in abaqusConfig.iterrows():
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
        # Append strain_range to strain_array
        truePlasticStrain = np.concatenate((truePlasticStrain, strain_range))


    #print(paramConfig)

    ###########################
    # Information declaration #
    ###########################

    info = {
        'projectPath': projectPath,
        'logPath': logPath,
        'resultPath': resultPath,
        'simPath': simPath,
        'targetPath': targetPath,
        'templatePath': templatePath,
        'optimizeStrategy': optimizeStrategy,
        'runInitialSims': runInitialSims,
        'numberOfInitialSims': numberOfInitialSims,
        'initialSimsSpacing': initialSimsSpacing,
        'material': material,
        'optimizerName': optimizerName,
        'hardeningLaw': hardeningLaw,
        'paramConfig': paramConfig,
        'geometry': geometry,
        'deviationPercent': deviationPercent,
        'strainStart': strainStart,
        'strainEnd': strainEnd,
        'strainStep': strainStep,
        'truePlasticStrain': truePlasticStrain,
    }

  
    ###############################################
    #  Printing the configurations to the console #
    ###############################################

    printLog(f"\nWelcome to the Abaqus parameter calibration project\n\n", logPath)
    printLog(f"The configurations you have chosen: \n", logPath)
    
    logTable = PrettyTable()

    logTable.field_names = ["Global Configs", "User choice"]
    logTable.add_row(["Optimize Strategy", optimizeStrategy])
    logTable.add_row(["Material", material])
    logTable.add_row(["Optimizer", optimizerName])
    logTable.add_row(["Hardening Law", hardeningLaw])
    logTable.add_row(["Geometry", geometry])
    logTable.add_row(["Deviation Percent", deviationPercent])
    logTable.add_row(["Run Initial Sims", runInitialSims])
    logTable.add_row(["Number of Initial Sims", numberOfInitialSims])

    printLog(logTable.get_string() + "\n", logPath)

    printLog("Generating necessary directories\n", logPath)
    printLog(f"The path to your main project folder is\n", logPath)
    printLog(f"{projectPath}\n", logPath)

    #############################
    # Returning the information #
    # ###########################

    return info

import os
import time
import pandas as pd
import numpy as np
from time import sleep
from prettytable import PrettyTable
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
    abaqusConfig = pd.read_excel("configs/abaqus_config.xlsx", nrows=1, engine="openpyxl")
    abaqusConfig = abaqusConfig.T.to_dict()[0]
    strainStart = float(abaqusConfig["strainStart"])
    strainEnd = float(abaqusConfig["strainEnd"])
    strainStep = float(abaqusConfig["strainStep"])

    #print(paramConfig)

    # The project path folder
    projectPath = os.getcwd()
    # The logging path
    logPath = f"log/{material}_{geometry}_{optimizerName}.txt"

    # The results path
    resultPath = f"results/{material}/{geometry}"

    # The simulations path
    simPath = f"simulations/{material}/{geometry}"

    # The templates path
    templatePath = f"templates/{material}/{geometry}"

    # The target path
    targetPath = f"targets/{material}/{geometry}"

    #########################################################
    # Creating necessary directories for the configurations #
    #########################################################

    def checkCreate(path):
        if not os.path.exists(path):
            os.mkdir(path)

    # For configs
    checkCreate("configs")

    # For log
    checkCreate("log")

    # For results 
    checkCreate("results")
    path = f"results/{material}"
    checkCreate(path)
    checkCreate(f"{path}/{geometry}")
    checkCreate(f"{path}/{geometry}/initial")
    checkCreate(f"{path}/{geometry}/iteration")

    # For simulations
    checkCreate("simulations")
    path = f"simulations/{material}"
    checkCreate(path)
    checkCreate(f"{path}/{geometry}")
    checkCreate(f"{path}/{geometry}/initial")
    checkCreate(f"{path}/{geometry}/iteration")

    # For templates
    checkCreate("templates")
    path = f"templates/{material}"
    checkCreate(path)
    checkCreate(f"{path}/{geometry}")

    # For targets
    checkCreate("targets")
    path = f"targets/{material}"
    checkCreate(path)
    checkCreate(f"{path}/{geometry}")

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

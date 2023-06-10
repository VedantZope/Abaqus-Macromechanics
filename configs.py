import os
import time
import pandas as pd
import numpy as np
from time import sleep
from prettytable import PrettyTable
from modules.helper import *
import copy

############################################################
#                                                          #
#          AUTOMATED MASS RVE GENERATION SOFTWARE          #
#   Tools required: Dream3D and Finnish Supercomputer CSC  #
#                                                          #
############################################################

# ------------------------------------#
#   Stage 0: Recording configurations #
# ------------------------------------#

def main_config():

    #########################
    # Global configurations #
    #########################

    globalConfig = pd.read_excel("configs/global_config.xlsx", nrows= 1, engine="openpyxl")
    globalConfig = globalConfig.T.to_dict()[0]
    #print(globalConfig)
    material = globalConfig["material"]
    
    numberOfRVE = globalConfig["numberOfRVE"]

    simulationIO = globalConfig["simulationIO"]

    skipSimulationRVEGeneration = globalConfig["skipSimulationRVEGeneration"]

    # The project path folder
    projectPath = os.getcwd()
    # The logging path
    logPath = f"log/{material}.txt"

    # The results path
    resultPath = f"results/{material}"

    # The simulations path
    simPath = f"simulations/{material}"

    # The templates path
    templatePath = f"templates/{material}"

    # The target path
    targetPath = f"targets/{material}"

    ###############################
    # Group of RVE configurations #
    ###############################

    RVEgroups = pd.read_excel("configs/RVE_groups.xlsx", engine="openpyxl")
 
    # Properties are column names of the RVE groups
    properties = list(RVEgroups.columns)
    #print(properties)
    #time.sleep(180)
    # Convert the DataFrame to a Python dictionary based on RVE group
    RVEgroups = RVEgroups.set_index('Group').to_dict(orient='index')
    RVEgroupsUnparsed = copy.deepcopy(RVEgroups)
    
    # Delete the details about NumFeatures
    # for groupIndex in RVEgroupsUnparsed:
    #     del RVEgroupsUnparsed[groupIndex]['NumFeaturesReference']
    #     del RVEgroupsUnparsed[groupIndex]['NumFeaturesType']
    #     del RVEgroupsUnparsed[groupIndex]['NumFeaturesEstimation']
    #print(RVEgroupsUnparsed)

    #print(RVEgroups)
    #time.sleep(180)
    for groupIndex in RVEgroups:
        RVEgroups[groupIndex]['Dimensions'] = parseDimensions(RVEgroups[groupIndex]['Dimensions'])
        RVEgroups[groupIndex]['Resolution'] = parseResolution(RVEgroups[groupIndex]['Resolution'])
        RVEgroups[groupIndex]['Origin'] = parseOrigin(RVEgroups[groupIndex]['Origin'])

    # print(RVEgroupsUnparsed)    
    # # Print the dictionary
    # print(RVEgroups)
    # sleep(180)

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
    
    # For simulations
    checkCreate("simulations")
    path = f"simulations/{material}"
    checkCreate(path)

    # For templates
    checkCreate("templates")
    path = f"templates/{material}"
    checkCreate(path)
    checkCreate(f"{path}/postProc")

    # For targets
    checkCreate("targets")
    path = f"targets/{material}"
    checkCreate(path)

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
        'material': material,
        'numberOfRVE': numberOfRVE,
        'simulationIO': simulationIO,
        'skipSimulationRVEGeneration': skipSimulationRVEGeneration,
        'RVEgroupsUnparsed': RVEgroupsUnparsed,
        'RVEgroups': RVEgroups,
        'properties': properties
    }

  
    ###############################################
    #  Printing the configurations to the console #
    ###############################################

    printLog(f"\nWelcome to the RVE generation software\n\n", logPath)
    printLog(f"The configurations you have chosen: \n", logPath)
    
    logTable = PrettyTable()

    logTable.field_names = ["Global Configs", "User choice"]
    logTable.add_row(["Material", material])
    logTable.add_row(["Number of group settings", len(RVEgroups)])
    logTable.add_row(["Number of RVEs each group", numberOfRVE])
    logTable.add_row(["Total number of RVE sims", len(RVEgroups) * numberOfRVE])
    logTable.add_row(["Simulation IO", simulationIO])
    printLog(logTable.get_string() + "\n", logPath)

    printLog("Generating necessary directories\n", logPath)
    printLog(f"The path to your main project folder is\n", logPath)
    printLog(f"{projectPath}\n", logPath)

    #############################
    # Returning the information #
    # ###########################

    return info

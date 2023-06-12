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

    globalConfig = pd.read_excel("configs/global_config.xlsx", nrows= 1, engine="openpyxl")
    globalConfig = globalConfig.T.to_dict()[0]

    material = globalConfig["material"]
    optimizerName = globalConfig["optimizerName"]

    # The project path folder
    projectPath = os.getcwd()
    # The logging path
    logPath = f"log/{material}_{optimizerName}.txt"

    # The results path
    resultPath = f"results/{material}"

    # The simulations path
    simPath = f"simulations/{material}"

    # The templates path
    templatePath = f"templates/{material}"

    # The target path
    targetPath = f"targets/{material}"

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
        'optimizerName': optimizerName
    }

  
    ###############################################
    #  Printing the configurations to the console #
    ###############################################

    printLog(f"\nWelcome to the Abaqus parameter calibration project\n\n", logPath)
    printLog(f"The configurations you have chosen: \n", logPath)
    
    logTable = PrettyTable()

    logTable.field_names = ["Global Configs", "User choice"]
    logTable.add_row(["Material", material])
    logTable.add_row(["Optimizer", optimizerName])
    printLog(logTable.get_string() + "\n", logPath)

    printLog("Generating necessary directories\n", logPath)
    printLog(f"The path to your main project folder is\n", logPath)
    printLog(f"{projectPath}\n", logPath)

    #############################
    # Returning the information #
    # ###########################

    return info

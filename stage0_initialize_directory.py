import os
import pandas as pd

#########################################################
# Creating necessary directories for the configurations #
#########################################################

def checkCreate(path):
    if not os.path.exists(path):
        os.makedirs(path)

def initialize_directory(optimizeStrategy, material, hardeningLaw, geometry, curveIndex):

    if optimizeStrategy == "SOO":
        # For log
        checkCreate("log")
        
        # For paramInfo
        path = f"SOO_paramInfo/{material}_{hardeningLaw}_{geometry}_curve{curveIndex}"
        checkCreate(path)

        # For results 
        path = f"SOO_results/{material}_{hardeningLaw}_{geometry}_curve{curveIndex}"
        checkCreate(path)
        checkCreate(f"{path}/initial")
        checkCreate(f"{path}/initial/common")
        checkCreate(f"{path}/iteration")
        checkCreate(f"{path}/iteration/common")

        # For simulations
        path = f"SOO_simulations/{material}_{hardeningLaw}_{geometry}_curve{curveIndex}"
        checkCreate(path)
        checkCreate(f"{path}/initial")
        checkCreate(f"{path}/iteration")

        # For targets
        path = f"SOO_targets/{material}_{hardeningLaw}_{geometry}_curve{curveIndex}"
        checkCreate(path)

        # For templates
        path = f"templates/{material}_{geometry}"
        checkCreate(path)

    elif optimizeStrategy == "MOO":
        geometryList = geometry
        # For log
        checkCreate("log")

        # For results 
        path = f"MOO_results/{material}_{hardeningLaw}"
        checkCreate(path)
        for geometry in geometryList:
            checkCreate(f"{path}/{geometry}/initial")
            checkCreate(f"{path}/{geometry}/initial/common")
            checkCreate(f"{path}/{geometry}/iteration")
            checkCreate(f"{path}/{geometry}/iteration/common")

        # For simulations
        path = f"MOO_simulations/{material}_{hardeningLaw}"
        checkCreate(path)
        for geometry in geometryList:
            checkCreate(f"{path}/{geometry}/initial")
            checkCreate(f"{path}/{geometry}/iteration")

        # For templates
        path = f"templates/{material}"
        checkCreate(path)
        for geometry in geometryList:
            checkCreate(f"{path}/{geometry}")

        # For targets
        path = f"MOO_targets/{material}"
        checkCreate(path)
        for geometry in geometryList:
            checkCreate(f"{path}/{geometry}")

    # The project path folder
    projectPath = os.getcwd()
    if optimizeStrategy == "SOO":
        # The logging path
        logPath = f"log/SOO_{material}_{hardeningLaw}_{geometry}_curve{curveIndex}.txt"
        # The paramInfo path
        paramInfoPath = f"SOO_paramInfo/{material}_{hardeningLaw}_{geometry}_curve{curveIndex}"
        # The results path
        resultPath = f"SOO_results/{material}_{hardeningLaw}_{geometry}_curve{curveIndex}"
        # The simulations path
        simPath = f"SOO_simulations/{material}_{hardeningLaw}_{geometry}_curve{curveIndex}"
        # The target path
        targetPath = f"SOO_targets/{material}_{hardeningLaw}_{geometry}_curve{curveIndex}"
        # The templates path
        templatePath = f"templates/{material}_{geometry}"
    elif optimizeStrategy == "MOO":
        # The logging path
        logPath = f"log/MOO_{material}_{hardeningLaw}.txt"
        # The paramInfo path
        paramInfoPath = f"MOO_paramInfo/{material}_{hardeningLaw}"
        # The results path
        resultPath = f"MOO_results/{material}"
        # The simulations path
        simPath = f"MOO_simulations/{material}"
        # The templates path
        templatePath = f"templates/{material}"
        # The target path
        targetPath = f"MOO_targets/{material}"

    return projectPath, logPath, paramInfoPath, resultPath, simPath, templatePath, targetPath

if __name__ == "__main__":
    globalConfig = pd.read_excel("configs/global_config.xlsx", nrows=1, engine="openpyxl")
    globalConfig = globalConfig.T.to_dict()[0]
    optimizeStrategy = globalConfig["optimizeStrategy"]
    material = globalConfig["material"]
    optimizerName = globalConfig["optimizerName"]
    hardeningLaw = globalConfig["hardeningLaw"]
    deviationPercent = globalConfig["deviationPercent"]
    geometry = globalConfig["geometry"]
    curveIndex = globalConfig["curveIndex"]
    numberOfInitialSims = globalConfig["numberOfInitialSims"]
    initialSimsSpacing = globalConfig["initialSimsSpacing"]
    initialize_directory(optimizeStrategy, material, hardeningLaw, geometry, curveIndex)
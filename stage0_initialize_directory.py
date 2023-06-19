import os


#########################################################
# Creating necessary directories for the configurations #
#########################################################

def checkCreate(path):
    if not os.path.exists(path):
        os.makedirs(path)

def initialize_directory(optimizeStrategy, material, geometry, hardeningLaw):

    if optimizeStrategy == "SOO":
        # For log
        checkCreate("SOO_log")

        # For results 
        path = f"SOO_results/{material}_{hardeningLaw}/{geometry}"
        checkCreate(path)
        checkCreate(f"{path}/initial")
        checkCreate(f"{path}/initial/common")
        checkCreate(f"{path}/iteration")
        checkCreate(f"{path}/iteration/common")

        # For simulations
        path = f"SOO_simulations/{material}_{hardeningLaw}/{geometry}"
        checkCreate(path)
        checkCreate(f"{path}/initial")
        checkCreate(f"{path}/iteration")

        # For templates
        path = f"templates/{material}/{geometry}"
        checkCreate(path)

        # For targets
        path = f"targets/{material}/{geometry}"
        checkCreate(path)
    
    elif optimizeStrategy == "MOO":
        geometryList = geometry
        # For log
        checkCreate("MOO_log")

        # For results 
        path = f"MOO_results/{material}_{hardeningLaw}"
        checkCreate(path)
        for geometry in geometryList:
            checkCreate(f"{path}/{geometry}/initial")
            checkCreate(f"{path}/{geometry}/initial/common")
            checkCreate(f"{path}/{geometry}/iteration")
            checkCreate(f"{path}/{geometry}/iteration/common")

        # For simulations
        path = f"SOO_simulations/{material}_{hardeningLaw}"
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
        path = f"targets/{material}"
        checkCreate(path)
        for geometry in geometryList:
            checkCreate(f"{path}/{geometry}")


    # The project path folder
    projectPath = os.getcwd()
    if optimizeStrategy == "SOO":
        # The logging path
        logPath = f"SOO_log/{material}_{hardeningLaw}_{geometry}.txt"
        # The results path
        resultPath = f"SOO_results/{material}_{hardeningLaw}/{geometry}"
        # The simulations path
        simPath = f"SOO_simulations/{material}_{hardeningLaw}/{geometry}"
        # The templates path
        templatePath = f"templates/{material}_{hardeningLaw}/{geometry}"
        # The target path
        targetPath = f"targets/{material}_{hardeningLaw}/{geometry}"
    elif optimizeStrategy == "MOO":
        # The logging path
        logPath = f"MOO_log/{material}_{hardeningLaw}.txt"
        # The results path
        resultPath = f"MOO_results/{material}"
        # The simulations path
        simPath = f"MOO_simulations/{material}"
        # The templates path
        templatePath = f"templates/{material}"
        # The target path
        targetPath = f"targets/{material}"

    return projectPath, logPath, resultPath, simPath, templatePath, targetPath
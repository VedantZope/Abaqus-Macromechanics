import pandas as pd
import numpy as np
import subprocess
import os
import matplotlib.pyplot as mp
from modules.hardeningLaws import *
from modules.helper import *
import sys
import shutil
import random
import time

class MOO_SIM():
    def __init__(self, info):
        self.info = info
   
    def latin_hypercube_sampling(self):
        paramConfig = self.info["paramConfig"]
        numberOfInitialSims = self.info["numberOfInitialSims"]
        linspaceValues = {}
        for param in paramConfig:
            linspaceValues[param] = np.linspace(
                start=paramConfig[param]["lowerBound"] * paramConfig[param]["exponent"], 
                stop=paramConfig[param]["upperBound"] * paramConfig[param]["exponent"], 
                num = self.info["initialSimsSpacing"])
            linspaceValues[param] = linspaceValues[param].tolist()   
        points = []
        for _ in range(numberOfInitialSims):
            while True:
                candidateParam = {}
                for param in linspaceValues:
                    random.shuffle(linspaceValues[param])
                    candidateParam[param] = linspaceValues[param].pop()
                if candidateParam not in points:
                    break
            points.append(candidateParam)

        return points

    def SOO_run_initial_simulations(self):
        indexParamsDict = self.SOO_preprocess_simulations_initial()
        self.SOO_write_paths_initial()
        self.SOO_submit_array_jobs_initial()
        self.SOO_postprocess_results_initial(indexParamsDict)

    def SOO_preprocess_simulations_initial(self):
        resultPath = self.info['resultPath']
        simPath = self.info['simPath']
        templatePath = self.info['templatePath'] 
        hardeningLaw = self.info['hardeningLaw']
        numberOfInitialSims = self.info['numberOfInitialSims']
        truePlasticStrain = self.info['truePlasticStrain']
        maxTargetDisplacement = self.info['maxTargetDisplacement']

        initial_params = self.latin_hypercube_sampling()
        #print(initial_params)
        np.save(f"{resultPath}/initial/common/parameters.npy", initial_params)
        initial_params = np.load(f"{resultPath}/initial/common/parameters.npy", allow_pickle=True).tolist()
        # Initializing the flow curves and force-displacement curves
        # The structure of flow curve: dict of (hardening law params typle) -> {stress: stressArray , strain: strainArray}
        
        flowCurves = {}
        
        for paramDict in initial_params:
            paramsTuple = tuple(paramDict.items())
            trueStress = calculate_flowCurve(paramDict, hardeningLaw, truePlasticStrain)
            flowCurves[paramsTuple] = {}
            flowCurves[paramsTuple]['strain'] = truePlasticStrain
            flowCurves[paramsTuple]['stress'] = trueStress
        np.save(f"{resultPath}/initial/common/flowCurves.npy", flowCurves)
        #print(flowCurves)

        indexParamsDict = {} # Map simulation folder index to the corresponding hardening law parameters
        for index, paramDict in enumerate(initial_params):
            indexParamsDict[str(index+1)] = tuple(paramDict.items())
        
        #print(simulationDict)
        # Copying the template folder to the simulation folder for the number of simulations
        for index in range(1, numberOfInitialSims + 1):
            # Create the simulation folder if not exists, else delete the folder and create a new one
            if os.path.exists(f"{simPath}/initial/{index}"):
                shutil.rmtree(f"{simPath}/initial/{index}")
            shutil.copytree(templatePath, f"{simPath}/initial/{index}")
            paramsTuple = indexParamsDict[str(index)]
            truePlasticStrain = flowCurves[paramsTuple]['strain']
            trueStress = flowCurves[paramsTuple]['stress']
            self.replace_flowCurve_material_inp(f"{simPath}/initial/{index}/material.inp", truePlasticStrain, trueStress)
            self.replace_maxDisp_geometry_inp(f"{simPath}/initial/{index}/geometry.inp", maxTargetDisplacement)
            self.replace_materialName_geometry_inp(f"{simPath}/initial/{index}/geometry.inp", "material.inp")
            self.create_parameter_file(f"{simPath}/initial/{index}", dict(paramsTuple))
            self.create_flowCurve_file(f"{simPath}/initial/{index}", truePlasticStrain, trueStress)
        return indexParamsDict

   
    
    def SOO_write_paths_initial(self):
        numberOfInitialSims = self.info['numberOfInitialSims']
        projectPath = self.info['projectPath']
        simPath = self.info['simPath']
        with open("linux_slurm/array_initial_file.txt", 'w') as filename:
            for index in range(1, numberOfInitialSims + 1):
                filename.write(f"{projectPath}/{simPath}/initial/{index}\n")
    
    def SOO_submit_array_jobs_initial(self):
        logPath = self.info['logPath']        
        numberOfInitialSims = self.info['numberOfInitialSims']
        printLog("Initial simulation preprocessing stage starts", logPath)
        printLog(f"Number of jobs required: {numberOfInitialSims}", logPath)
        subprocess.run(f"sbatch --wait --array=1-{numberOfInitialSims} linux_slurm/puhti_abaqus_array.sh", shell=True)
        printLog("Initial simulation postprocessing stage finished", logPath)
    
    def SOO_postprocess_results_initial(self, indexParamsDict):
        numberOfInitialSims = self.info['numberOfInitialSims']
        simPath = self.info['simPath']
        resultPath = self.info['resultPath']
        logPath = self.info['logPath']
        
        # The structure of force-displacement curve: dict of (hardening law params typle) -> {force: forceArray , displacement: displacementArray}

        FD_Curves = {}
        for index in range(1, numberOfInitialSims + 1):
            if not os.path.exists(f"{resultPath}/initial/{index}"):
                os.mkdir(f"{resultPath}/initial/{index}")
            shutil.copy(f"{simPath}/initial/{index}/FD_Curve.txt", f"{resultPath}/initial/{index}")
            shutil.copy(f"{simPath}/initial/{index}/FD_Curve_Plot.tif", f"{resultPath}/initial/{index}")
            shutil.copy(f"{simPath}/initial/{index}/Deformed_Specimen.tif", f"{resultPath}/initial/{index}")
            shutil.copy(f"{simPath}/initial/{index}/parameters.xlsx", f"{resultPath}/initial/{index}")
            shutil.copy(f"{simPath}/initial/{index}/parameters.csv", f"{resultPath}/initial/{index}")
            shutil.copy(f"{simPath}/initial/{index}/flowCurve.xlsx", f"{resultPath}/initial/{index}")
            shutil.copy(f"{simPath}/initial/{index}/flowCurve.csv", f"{resultPath}/initial/{index}")
                        
            paramsTuple = indexParamsDict[str(index)]
            displacement, force = read_FD_Curve(f"{simPath}/initial/{index}/FD_Curve.txt")
            FD_Curves[paramsTuple] = {}
            FD_Curves[paramsTuple]['displacement'] = displacement
            FD_Curves[paramsTuple]['force'] = force
            create_FD_Curve_file(f"{resultPath}/initial/{index}", displacement, force)
            
        # Returning force-displacement curve data
        np.save(f"{resultPath}/initial/common/FD_Curves.npy", FD_Curves)
        printLog("Saving successfully all simulation results", logPath)

    def SOO_run_iteration_simulations(self, paramsDict):
        self.SOO_preprocess_simulations_iteration(paramsDict)
        self.SOO_write_paths_iteration()
        self.SOO_submit_array_jobs_iteration()
        parameters, FD_Curves, flowCurves = self.SOO_postprocess_results_iteration(paramsDict)
        return parameters, FD_Curves, flowCurves
    
    def SOO_preprocess_simulations_iteration(self, paramsDict):
        resultPath = self.info['resultPath']
        simPath = self.info['simPath']
        templatePath = self.info['templatePath'] 
        hardeningLaw = self.info['hardeningLaw']
        numberOfInitialSims = self.info['numberOfInitialSims']
        truePlasticStrain = self.info['truePlasticStrain']
        maxTargetDisplacement = self.info['maxTargetDisplacement']

        flowCurves = {}
        
        paramsTuple = tuple(paramDict.items())
        trueStress = calculate_flowCurve(paramDict, hardeningLaw, truePlasticStrain)
        flowCurves[paramsTuple] = {}
        flowCurves[paramsTuple]['strain'] = truePlasticStrain
        flowCurves[paramsTuple]['stress'] = trueStress
        np.save(f"{resultPath}/initial/common/flowCurves.npy", flowCurves)
        #print(flowCurves)
        
        #print(simulationDict)
        # Copying the template folder to the simulation folder for the number of simulations
        for index in range(1, numberOfInitialSims + 1):
            # Create the simulation folder if not exists, else delete the folder and create a new one
            if os.path.exists(f"{simPath}/initial/{index}"):
                shutil.rmtree(f"{simPath}/initial/{index}")
            shutil.copytree(templatePath, f"{simPath}/initial/{index}")
            paramsTuple = indexParamsDict[str(index)]
            truePlasticStrain = flowCurves[paramsTuple]['strain']
            trueStress = flowCurves[paramsTuple]['stress']
            replace_flowCurve_material_inp(f"{simPath}/initial/{index}/material.inp", truePlasticStrain, trueStress)
            replace_maxDisp_geometry_inp(f"{simPath}/initial/{index}/geometry.inp", maxTargetDisplacement)
            replace_materialName_geometry_inp(f"{simPath}/initial/{index}/geometry.inp", "material.inp")
            create_parameter_file(f"{simPath}/initial/{index}", dict(paramsTuple))
            create_flowCurve_file(f"{simPath}/initial/{index}", truePlasticStrain, trueStress)
        return indexParamsDict



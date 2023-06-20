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

class SIM():
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
        projectPath = self.info['projectPath']
        logPath = self.info['logPath']

        resultPath = self.info['resultPath']
        simPath = self.info['simPath']
        targetPath = self.info['targetPath']
        templatePath = self.info['templatePath'] 
        material = self.info['material']
        optimizeStrategy = self.info['optimizeStrategy']
        optimizerName = self.info['optimizerName']
        hardeningLaw = self.info['hardeningLaw']
        paramConfig = self.info['paramConfig']
        geometry = self.info['geometry']
        deviationPercent = self.info['deviationPercent']
        numberOfInitialSims = self.info['numberOfInitialSims']
        truePlasticStrain = self.info['truePlasticStrain']
        maxTargetDisplacement = self.info['maxTargetDisplacement']

        initial_params = self.latin_hypercube_sampling()
        #print(initial_params)
        np.save(f"{resultPath}/initial/common/initial_params.npy", initial_params)
        initial_params = np.load(f"{resultPath}/initial/common/initial_params.npy", allow_pickle=True).tolist()
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
            displacement, force = self.read_FD_Curve(f"{simPath}/initial/{index}/FD_Curve.txt")
            FD_Curves[paramsTuple] = {}
            FD_Curves[paramsTuple]['displacement'] = displacement
            FD_Curves[paramsTuple]['force'] = force
            self.create_FD_Curve_file(f"{resultPath}/initial/{index}", displacement, force)
            
        # Returning force-displacement curve data
        np.save(f"{resultPath}/initial/common/FD_Curves.npy", FD_Curves)
        printLog("Saving successfully all simulation results", logPath)

    def read_FD_Curve(self, filePath):
        output_data=np.loadtxt(filePath, skiprows=2)
        # column 1 is time step
        # column 2 is displacement
        # column 3 is force
        columns=['X', 'Displacement', 'Force']
        df = pd.DataFrame(data=output_data, columns=columns)
        displacement = df.iloc[:, 1].tolist()
        force = df.iloc[:, 2].tolist()
        return displacement, force

    def MOO_run_initial_simulations(self):
        pass

    def create_parameter_file(self, filePath, paramsDict):
        columns = ["Parameter", "Value"]
        df = pd.DataFrame(columns=columns)
        for key, value in paramsDict.items():
            df.loc[len(df.index)] = [key, value]
        df.to_excel(f"{filePath}/parameters.xlsx", index=False)
        df.to_csv(f"{filePath}/parameters.csv", index=False)

    def create_flowCurve_file(self, filePath, truePlasticStrain, trueStress):
        columns = ["strain,-", "stress,MPa", "stress,Pa"]
        df = pd.DataFrame(columns=columns)
        for i in range(len(truePlasticStrain)):
            df.loc[len(df.index)] = [truePlasticStrain[i], trueStress[i], trueStress[i]*1e6]
        df.to_excel(f"{filePath}/flowCurve.xlsx", index=False)
        df.to_csv(f"{filePath}/flowCurve.csv", index=False)
    
    def create_FD_Curve_file(self, filePath, displacement, force):
        columns = ["displacement,mm", "force,kN", "force,N"]
        df = pd.DataFrame(columns=columns)
        for i in range(len(displacement)):
            df.loc[len(df.index)] = [displacement[i], force[i] * 1e-3, force[i]]
        df.to_excel(f"{filePath}/FD_Curve.xlsx", index=False)
        df.to_csv(f"{filePath}/FD_Curve.csv", index=False)

    def replace_flowCurve_material_inp(self, filePath, truePlasticStrain, trueStress):
        with open(filePath, 'r') as material_inp:
            material_inp_content = material_inp.readlines()
        # Locate the section containing the stress-strain data
        start_line = None
        end_line = None
        for i, line in enumerate(material_inp_content):
            if '*Plastic' in line:
                start_line = i + 1
            elif '*Density' in line:
                end_line = i
                break

        if start_line is None or end_line is None:
            raise ValueError('Could not find the stress-strain data section')

        # Modify the stress-strain data
        new_stress_strain_data = zip(trueStress, truePlasticStrain)
        # Update the .inp file
        new_lines = []
        new_lines.extend(material_inp_content[:start_line])
        new_lines.extend([f'{stress},{strain}\n' for stress, strain in new_stress_strain_data])
        new_lines.extend(material_inp_content[end_line:])

        # Write the updated material.inp file
        with open(filePath, 'w') as file:
            file.writelines(new_lines)

    def replace_maxDisp_geometry_inp(self, filePath, maxTargetDisplacement):
        with open(filePath, 'r') as geometry_inp:
            geometry_inp_content = geometry_inp.readlines()
        start_line = None
        end_line = None
        for i, line in enumerate(geometry_inp_content[-60:]):
            if line.startswith('*Boundary, amplitude'):
                original_index = len(geometry_inp_content) - 60 + i
                start_line = original_index + 1
                end_line = original_index + 2
                break

        if start_line is None or end_line is None:
            raise ValueError('Could not find the *Boundary, amplitude displacement section')

        new_disp_data = f"Disp, 2, 2, {maxTargetDisplacement}\n"

        new_lines = []
        new_lines.extend(geometry_inp_content[:start_line])
        new_lines.extend([new_disp_data])
        new_lines.extend(geometry_inp_content[end_line:])

        with open(filePath, 'w') as file:
            file.writelines(new_lines)

    def replace_materialName_geometry_inp(self, filePath, materialName):
        with open(filePath, 'r') as geometry_inp:
            geometry_inp_content = geometry_inp.readlines()
        start_line = None
        end_line = None
        for i, line in enumerate(geometry_inp_content[-60:]):
            if line.startswith('*INCLUDE, INPUT='):
                original_index = len(geometry_inp_content) - 60 + i
                start_line = original_index
                end_line = original_index + 1
                break

        if start_line is None or end_line is None:
            raise ValueError('Could not find the **INCLUDE, INPUT= section')

        new_material_data = f"*INCLUDE, INPUT={materialName}\n"

        new_lines = []
        new_lines.extend(geometry_inp_content[:start_line])
        new_lines.extend([new_material_data])
        new_lines.extend(geometry_inp_content[end_line:])

        with open(filePath, 'w') as file:
            file.writelines(new_lines)

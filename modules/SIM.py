import pandas as pd
import numpy as np
import subprocess
import os
import matplotlib.pyplot as mp
from modules.hardeningLaws import *
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
                start=paramConfig[param]["lowerBound"], 
                stop=paramConfig[param]["upperBound"], 
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
        runInitialSims = self.info['runInitialSims']
        numberOfInitialSims = self.info['numberOfInitialSims']
        strainStart = self.info['strainStart']
        strainEnd = self.info['strainEnd']
        strainStep = self.info['strainStep']
        truePlasticStrain = self.info['truePlasticStrain']

        initial_params = self.latin_hypercube_sampling()
        #print(initial_params)
        np.save(f"{resultPath}/initial/common/initial_params.npy", initial_params)
        
        # Initializing the flow curves and force-displacement curves
        # The structure of flow curve: dict of (hardening law params typle) -> {stress: stressArray , strain: strainArray}
        # The structure of force-displacement curve: dict of (hardening law params typle) -> {force: forceArray , displacement: displacementArray}
        
        flowCurves = {}
        FDCurves = {}
        for paramDict in initial_params:
            paramTuple = tuple(paramDict.items())
            if hardeningLaw == "Swift":
                c1, c2, c3 = paramDict['c1'], paramDict['c2'], paramDict['c3']
                flowCurve = Swift(c1, c2, c3, truePlasticStrain)
            if hardeningLaw == "SwiftVoce":
                c1, c2, c3, c4, c5, c6, c7 = paramDict['c1'], paramDict['c2'], paramDict['c3'], paramDict['c4'], paramDict['c5'], paramDict['c6'], paramDict['c7']
                flowCurve = SwiftVoce(c1, c2, c3, c4, c5, c6, c7, truePlasticStrain)
            FDCurves[paramTuple] = {}
            flowCurves[paramTuple] = {}
            flowCurves[paramTuple]['stress'] = flowCurve
            flowCurves[paramTuple]['strain'] = truePlasticStrain
        np.save(f"{resultPath}/initial/common/flowCurves.npy", flowCurves)
        print(flowCurves)

        # Copying the template folder to the simulation folder for the number of simulations
        for index in range(1, numberOfInitialSims + 1):
            # Create the simulation folder if not exists, else delete the folder and create a new one
            if os.path.exists(f"{simPath}/initial/{index}"):
                shutil.rmtree(f"{simPath}/initial/{index}")
            shutil.copytree(templatePath, f"{simPath}/initial/{index}")
            self.replace_flowCurve_

        print("Hello")
        time.sleep(30)

    def replace_flowCurve_material_inp(self, filePath, truePlasticStrain):
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

        stress_strain_lines = material_inp_content[start_line:end_line]
        stress_strain_data = []
        for line in stress_strain_lines:
            data = line.split(',')  # Adjust delimiter based on your file format
            stress_strain_data.append((float(data[0]), float(data[1])))

        # Step 4: Modify the stress-strain data
        new_stress_strain_data = zip(trueStress, trueStrain)

        # Step 5: Update the .inp file
        new_lines = []
        new_lines.extend(material_inp_content[:start_line])
        new_lines.extend([f'{stress},{strain}\n' for stress, strain in new_stress_strain_data])
        new_lines.extend(material_inp_content[end_line:])

        # Step 6: Write the updated .inp file
        with open('Material_DP1000_Mises.inp', 'w') as file:
            file.writelines(new_lines)

    def MOO_run_initial_simulations(self):
        pass
    


    def run_simulation(self):
        # Run the simulation
        process = subprocess.Popen([batch_file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

        # Read and print the output line by line in real-time
        for line in iter(process.stdout.readline, ''):
            print("Output:", line, end='')

        # Read and print the error messages line by line in real-time
        for line in iter(process.stderr.readline, ''):
            print("Error:", line, end='')

        # Wait for the process to finish
        process.wait()

        # Get the final return code
        return_code = process.returncode

        # Print the return code
        print("Return code:", return_code)

    def run_iteration_simulation(self, trueStress, truePlasticStrain):
        # Copy template folder to simulation folder
        simPath = info['simulationPath']
        templatePath = info['templatePath']
        shutil.copytree(templatePath/{}, simPath)
        
        #===================updating the material.inp file====================
        material_inp_file_path = f"{geometry}/DP1000_Mises.inp"
        batch_file_path = f"submit-postprocess.bat"

        #=================execute the simulation and post processs its results=================
        #run a batch file with command: 
        #(abaqus job=jobname.inp interactive cpus=6
        #abaqus cae noGUI=postprocess.py)

        self.run_simulation()


        # Open the output file in read mode
        with open(f"{working_directory}/texts/F-D_data.txt", 'r') as file:
            # Read all the lines from the file
            lines = file.readlines()

        # Remove the first two rows
        new_lines = lines[2:]

        # Open the file in write mode to overwrite the contents
        with open(f"{working_directory}/texts/output.txt", 'w') as file:
            # Write the modified lines back to the file
            file.writelines(new_lines)

        output_data=np.loadtxt(f"{working_directory}/texts/output.txt")
        #column 1 is time
        #column 2 is disp
        #column 3 is force


        output_data = pd.DataFrame(data=output_data)
        columns=['X', 'Displacement', 'Force']


        x_new = output_data.iloc[:, 1].tolist()
        y_new = output_data.iloc[:, 2].tolist()
        
        # Returning force-displacement curve data
        return x_new, y_new





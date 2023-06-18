
import pandas as pd
import numpy as np
import subprocess
import os
import matplotlib.pyplot as mp
import sys
import shutil



def SIM(info):
    def __init__(self, info):
        self.info = info



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

        with open(inp_file_path, 'r') as inp_file:
            inp_content = inp_file.readlines()

        # Locate the section containing the stress-strain data
        start_line = None
        end_line = None
        for i, line in enumerate(inp_content):
            if '*Plastic' in line:
                start_line = i + 1
            elif '*Density' in line:
                end_line = i
                break

        if start_line is None or end_line is None:
            raise ValueError('Could not find the stress-strain data section')

        stress_strain_lines = inp_content[start_line:end_line]
        stress_strain_data = []
        for line in stress_strain_lines:
            data = line.split(',')  # Adjust delimiter based on your file format
            stress_strain_data.append((float(data[0]), float(data[1])))

        # Step 4: Modify the stress-strain data
        new_stress_strain_data = zip(trueStress, trueStrain)

        # Step 5: Update the .inp file
        new_lines = []
        new_lines.extend(inp_content[:start_line])
        new_lines.extend([f'{stress},{strain}\n' for stress, strain in new_stress_strain_data])
        new_lines.extend(inp_content[end_line:])

        # Step 6: Write the updated .inp file
        with open('Material_DP1000_Mises.inp', 'w') as file:
            file.writelines(new_lines)


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





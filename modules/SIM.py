import mpmath
import pandas as pd
import numpy as np
import subprocess
import os
import matplotlib.pyplot as mp
import sys


working_directory = os.getcwd()
inp_file_path = f"{working_directory}\Material_DP1000_Mises.inp"
batch_file_path = f"{working_directory}\submit-postprocess.bat"

#================defining the swift law=============================
def swift_law(c1,c2,c3,ε):
    return c1 * mpmath.exp(c3 * mpmath.log(c2 + ε))


def get_xy(paramc1, paramc2, paramc3):

    #===========creating the material input data usin 3 params swift equation============
    start = 0.00
    end = 0.053
    step = 0.0002 #increament in strain
    strain_list = []
    stress_list = []
    current_value = float(start)

    #creating strain points
    while current_value <= end:
        strain_list.append(round(current_value, 4))  # Round to 4 decimal places
        current_value += step
        
    #creating stress points
    for i in strain_list:
            stress_list.append((swift_law(paramc1,paramc2,paramc3,i)))


    #===================updating the material.inp file====================

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
    new_stress_strain_data = zip(stress_list, strain_list)

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

    # Execute the batch file
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


    # Open the output file in read mode
    with open(f"{working_directory}/Exported_Texts/F-D_data.txt", 'r') as file:
        # Read all the lines from the file
        lines = file.readlines()

    # Remove the first two rows
    new_lines = lines[2:]

    # Open the file in write mode to overwrite the contents
    with open(f"{working_directory}/Exported_Texts/output.txt", 'w') as file:
        # Write the modified lines back to the file
        file.writelines(new_lines)

    output_data=np.loadtxt(f"{working_directory}/Exported_Texts/output.txt")
    #column 1 is time
    #column 2 is disp
    #column 3 is force


    output_data = pd.DataFrame(data=output_data)
    columns=['X', 'Displacement', 'Force']


    x_new = output_data.iloc[:, 1].tolist()
    y_new = output_data.iloc[:, 2].tolist()
    
    # Returning force-displacement curve data
    return x_new, y_new





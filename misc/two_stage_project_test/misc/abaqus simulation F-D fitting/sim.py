#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#name of file sim.py


# In[ ]:


import pandas as pd
import numpy as np
import subprocess
import os
import matplotlib.pyplot as mp
import sys


# In[ ]:


#=================path of the files=====================
inp_file_path = 'Material_DP1000_Mises.inp'
batch_file_path = r"C:\SIMULIA\temp\fully automated\run-postprocess.bat"
working_directory = r"C:\SIMULIA\temp\fully automated"
output_directory = r""


# In[57]:


#================defining the swift law=============================
def swift_law(c1,c2,c3,ε):
    σ=c1*(c2+ε)**c3 #3 parameter 
    return(σ)


# In[ ]:


def get_xy(paramc1,paramc2,paramc3):

    #===========creating the material input data usin 3 params swift equation============

    start = 0.00
    end = 0.05
    step = 0.0002 #increament in strain

    strain_list = []
    stress_list = []

    current_value = float(start)

    #creating strain pints
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


    # Change the current working directory
    os.chdir(working_directory)

    result = subprocess.run([batch_file_path], capture_output=True, text=True, shell=True)

    # Access the output and error messages
    output = result.stdout
    errors = result.stderr

    # Print the output and errors
    print("Output:", output)
    print("Errors:", errors)


    # Open the output file in read mode
    with open(output_directory+'F-D data.txt', 'r') as file:
        # Read all the lines from the file
        lines = file.readlines()

    # Remove the first two rows
    new_lines = lines[2:]

    # Open the file in write mode to overwrite the contents
    with open(output_directory+'output.txt', 'w') as file:
        # Write the modified lines back to the file
        file.writelines(new_lines)

    output_data=np.loadtxt('output.txt')
    #column 1 is time
    #column 2 is disp
    #column 3 is force


    output_data = pd.DataFrame(data=output_data)
    #columns=['X', 'Displacement', 'Force']


    x_new = output_data.iloc[:, 1].tolist()
    y_new = output_data.iloc[:, 2].tolist()

    return x_new, y_new


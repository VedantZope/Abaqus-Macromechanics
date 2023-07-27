
# Swift Voce Hardening Parameter Calibration Project with Abaqus

This project code is used to optimize the parameters in the Swift and Swift-Voce hardening laws.

`Problem formulation`: Given an experimental force displacement (FD) curve of a geometry design (such as Notched Dog Bone NDBR50) of a material (such as DP1000) under a certain temperature (400 degrees), determine its flow curve (a type of true stress-true strain curve but without elastic region) such that when this flow curve is fed as input to Abaqus, the software simulation will produce a FD curve that matches the experimental FD curve. 

`Question`: why is optimizing a flow curve related to optimizing the parameters in the Swift Voce hardening law?

`Answer`: If we calibrate the flow curve directly, we need to calculate individual points on the flow curve, which could be over 500 points and the task can become challenging (may require a neural networks). The Swift Voce equation is a parametric representation of the flow curve, which is capable of producing any flow curve with high precision given only 7 parameters (instead of 500 like above!). Therefore, calibrating the Swift Voce parameters is equivalent to calibrating the flow curve, just that we only need to calibrate far fewer number of unknown variables.   

`Workflow`: This project is divided into 2 main workflows: the single objective optimization (SOO) task which requires calibrating the flow curve to match only one FD curve. Another is multiple objective optimization (MOO) task, which requires calibrating the flow curve to match many FD curves of different geometries at once in the same temperature. 

`Stages`: This project has 2 main stages (both for SOO and MOO): producing a large number of initial guesses as the knowledge basis for optimization, and running iterative calibration, where Bayesian optimization continues to update its surrogate model (Gaussian process) as more simulations are added to the knowledge basis.

## Status

![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)
![Contributors](https://img.shields.io/github/contributors/springnuance/Abaqus-Macromechanics-Project.svg)
![Number of Commits](https://img.shields.io/github/commit-activity/y/springnuance/Abaqus-Macromechanics-Project.svg)

## Authors

- [@VedantZope](https://www.github.com/VedantZope). Tasks: Problem formulation, writing Abaqus scripts, preparing simulation templates, running simulations and fine tuning results
- [@SpringNuance](https://www.github.com/springnuance) (Xuan Binh). Tasks: Optimization strategy, workflow design and programming project code. 

## Acknowledgements

 - [Professor Junhe Lian](https://scholar.google.com/citations?user=HO6x8pkAAAAJ&hl=en) for supervising this project
 - [Doctor Li Zinan](https://www.researchgate.net/profile/Zinan-Li-2) for providing experimental input data
 - [Doctor Rongfei Juan](https://www.researchgate.net/profile/Rongfei-Juan) for providing tips on presentation details

## How to run the project code

The only command you would need to run the project code is 
python optimize.py

- Stage 1: Fixing the configs/global_configs.xlsx for your desire problem

- Stage 2: Running python stage0_initialize_directory.py. This is for folders generation

- Stage 3: If you use SOO, You need to create FD_curve.csv under directory SOO_targets\{material}_{hardeningLaw}_{geometry}_curve{curveIndex}
         This csv file should have 3 columns name displacement,mm force,kN and force,N
- Stage 4: You need to create paramInfo.xlsx under directory SOO_paramInfo\{material}_{hardeningLaw}_{geometry}_curve{curveIndex}
         This csv file should have N columns depending on the number of parameters in hardening law. Other columns like lowerBound, upperBound, exponent, name and unit should also be defined

- Stage 5: Drag the whole project code onto CSC and run
         cd projectDirectory
         module load python-data
         pip install --user requirements.txt
         python optimize.py
  
- Stage 6: The results will be output to the directory SOO_results\{material}_{hardeningLaw}_{geometry}_curve{curveIndex}\iteration
         Under this directory are iteration simulation result with index 1,2,3,... and a folder called common, which stores FD_Curves.npy and flowCurves.npy. 
         The data structure of FD_Curves.npy is as follows:
         dictionary of 
          keys: parameter tuple of hardening law (("c1": value), ("c2": value), ("c3": value))
          values: dictionary of force displacement
             key: force in N, value: numpy array
             key: displacement in mm, value: numpy array

Note: The workflow is resistant to interruption. When you interrupted at iteration N, and when you rerun again with the same configurations, it will resume at iteration N again. 
         
- Stage 7: The workflow already assumed that the initial guesses are smooth force-displacement curve.
         If FD curve produced by Abaqus is wavy, please use notebooks/smooth.ipynb to process the FD curves


# Swift Voce Hardening Parameter Calibration with Abaqus

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

At the basics, these are the only folders and files required for running simulations. Others are not required or for postprocessing only. 

```
Abaqus-Macromechanics-Project
├───configs <- This is for global and strain points configurations. Users need to manually change the global_configs.xlsx
├───linux_slurm <- Used for running array jobs of simulations on CSC platform
├───log <- Logging the progress into a text file
├───modules <- Containing python files as smaller modules that aid the calibration workflow
├───optimizers <- Containing Bayesian optimization code
├───MOO_paramInfo <- Defines Swift/Swift Voce parameter settings (range and exponent). Users need to manually define this
│   └───{material}_{hardeningLaw}_curve{curveIndex}
├───MOO_results <- Storing the simulation results
│   └───{material}_{hardeningLaw}_curve{curveIndex}
│       ├───{geometry}
│       │   ├───initial
│       │   │   ├───common
│       │   │   └───data
│       │   └───iteration
│       │       ├───common
│       │       └───data
├───MOO_simulations <- Temporary folders for storing simulations. After simulation, outputs are copied to result folder
│   └───{material}_{hardeningLaw}_curve{curveIndex}
│       ├───{geometry}
│       │   ├───initial
│       │   └───iteration
├───MOO_targets <- Storing experimental force-displacement curves that needs to be matched. Users need to manually add the FD curve excel files
│   ├───{material}_{hardeningLaw}_curve{curveIndex}
│   │   ├───{geometry}
├───SOO_paramInfo (Same as MOO)
│   ├───{material}_{hardeningLaw}_{geometry}_curve{curveIndex}
├───SOO_results (Same as MOO)
│   ├───{material}_{hardeningLaw}_{geometry}_curve{curveIndex}
│   │   ├───initial
│   │   │   ├───common
│   │   │   └───data
│   │   ├───iteration
│   │   │   ├───common
│   │   │   └───data
├───SOO_simulations (Same as MOO)
│   ├───{material}_{hardeningLaw}_{geometry}_curve{curveIndex}
│   │   ├───initial
│   │   └───iteration
├───SOO_targets (Same as MOO)
│   ├───{material}_{hardeningLaw}_{geometry}_curve{curveIndex}
├───templates <- Storing simulation templates of Abaqus. Users need to manually add the templates and postprocess.py
│   ├───{material}
│   │   └───{geometry}
├───stage0_configs.py <- Extracting the global configurations
├───stage0_initialize_directory.py <- Used to automatically initialize required project hierarchy folders
├───stage1_MOO_prepare_targetCurve.py <- Used to extract the FD curves from MOO_targets folder
├───stage2_MOO_run_initialSims.py <- Used to run initial guesses. Results are stored in MOO_results/{material}_{hardeningLaw}_curve{curveIndex}/{geometry}/initial
├───stage3_MOO_prepare_simCurves.py <- Used to extract initial guess simulations and iteration simulation (if they exist) and combine/interpolate the FD curves
├───stage4_MOO_iterative_calibration.py <- Used to run iterative parameter calibration. Results are stored in MOO_results/{material}_{hardeningLaw}_curve{curveIndex}/{geometry}/iteration
├───stage1_SOO_prepare_targetCurve.py (Same as MOO)
├───stage2_SOO_run_initialSims.py (Same as MOO)
├───stage3_SOO_prepare_simCurves.py (Same as MOO)
├───stage4_SOO_iterative_calibration.py (Same as MOO)
```

- Stage 1: Fixing the configs/global_configs.xlsx for your desire problem

- Stage 2: Running python stage0_initialize_directory.py. This is for folders generation

- Stage 3: You need to create FD_curve.csv under directory SOO_targets\{material}_{hardeningLaw}_{geometry}_curve{curveIndex}
         This csv file should have 3 columns name displacement,mm force,kN and force,N
- Stage 4: You need to create paramInfo.xlsx under directory SOO_paramInfo\{material}_{hardeningLaw}_{geometry}_curve{curveIndex}
         This csv file should have N columns depending on the number of parameters in hardening law. Other columns like lowerBound, upperBound, exponent, name and unit should also be defined

- Stage 5: Drag the whole project code onto CSC and run
  ```bash
  $ cd projectDirectory
  $ module load python-data
  $ pip install --user requirements.txt # Run only once
  $ python optimize.py # The only command you would need to run the project code
  ```
- Stage 6: The results will be output to the directory SOO_results\{material}_{hardeningLaw}_{geometry}_curve{curveIndex}\iteration
         Under this directory are iteration simulation result with index 1,2,3,... and a folder called common, which stores FD_Curves.npy and flowCurves.npy. 
         The data structure of FD_Curves.npy is as follows:

  dictionary of \
  &nbsp;&nbsp;&nbsp;&nbsp; keys: parameter tuple of hardening law (("c1": value), ("c2": value), ("c3": value)) \
  &nbsp;&nbsp;&nbsp;&nbsp; values: dictionary of force displacement \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; key: force in N, value: numpy array \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; key: displacement in mm, value: numpy array 

Note: The workflow is resistant to interruption. When you interrupted at iteration N, and when you rerun again with the same configurations, it will resume at iteration N again. 
         
- Stage 7: The workflow assumed that the initial guesses are wavy/jagged force-displacement curve.
         If FD curve produced by Abaqus is wavy, the code will use Savgol filter to smoothen the FD curves

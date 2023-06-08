Running the Bayesian Optimization Program
This is a guide for setting up and running the Bayesian optimization program, which uses experimental force displacement data for parametric evaluation. This guide outlines the function of each script, their interdependencies, and the steps needed to successfully run the program.

Scripts Overview
1. Disp-Force_ExpRT_ndb50.csv: This file contains the experimental force displacement data that needs to be fitted for parametric evaluation.

2. bayesopt_copy.py: This is the main script that implements the Bayesian optimization. It calculates the loss function based on the difference between the simulated data (generated using sim.py) and the experimental data.

3. sim.py: This script submits the job for simulation and waits for its completion. The progress of the simulation can be tracked on the terminal.

4. postprocess.py: This script post-processes the simulation data after the simulation job is run and returns it back to the Bayesian optimization script (bayesopt_copy.py) for calculating the loss function.

The Bayesian optimization process is iterative, meaning that it continues until the parameters are optimally determined or a specified number of iterations are completed.

Running the Program
Follow these steps to run the program:

Step 1: Setup the working directory

Both postprocess.py and sim.py have a working directory specified at the top of the file. You need to change this to the current working directory where these files are stored.

Step 2: Run the bayesopt_copy.py script

After setting up the working directories in postprocess.py and sim.py, you are ready to run the main script. In the terminal, navigate to your working directory and run the bayesopt_copy.py script:

python bayesopt_copy.py
The Bayesian optimization will run iteratively until the parameters are determined, or until the specified number of iterations are finished.


#!/usr/bin/env python
# coding: utf-8

# In[1]:



# In[56]:


# implement a dummy Bayesian optimization algorithm
import numpy as np
import pandas as pd
import sim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.interpolate import CubicSpline
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from sklearn.gaussian_process.kernels import RBF, Matern # you can try to import other kernels from sklearn as well


# ### Prepare target flow curve

# In[57]:


# Read the CSV file into a DataFrame (ground truth)
df = pd.read_csv('Disp-Force_ExpRT_ndb50.csv')
# print(df)
# Extract the true strain and average true stress columns
expt_Disp = df['Disp /mm'].to_numpy()
expt_Force = df['Force /kN'].to_numpy()


# In[58]:

# Continuous searching space
param_bounds = {
    "c1": (700, 1800),  
    "c2": (0.1 * 1e-14, 10 * 1e-14) ,    
    "c3": (0.001 , 0.1 ) 
}

# simulated F-D data
def simulated_FD(c1,c2,c3):
    x,y = sim.get_xy(c1,c2,c3)
    return np.array(x),np.array(y)

# Note: BO in Bayes-Opt tries to maximize, so you should use the inverse of the loss function.
def lossFunction( **solution):
    #print(solution)
    c1 = solution["c1"]
    c2 = solution["c2"] 
    c3 = solution["c3"]
    sim_Disp,sim_Force = simulated_FD(c1,c2,c3)
    # Sort simulated data by displacement (if not sorted already)
    sort_indices = np.argsort(sim_Disp)
    sim_Disp = sim_Disp[sort_indices]
    sim_Force = sim_Force[sort_indices]

    # Create a cubic spline interpolation function
    cubic_spline = CubicSpline(sim_Disp, sim_Force)

    # Evaluate the interpolated function at the x values of the experimental data
    interp_sim_Force = cubic_spline(expt_Disp)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(expt_Force, interp_sim_Force))
    return -rmse


# In[59]:


class BO():
    
    ##################################
    # OPTIMIZER CLASS INITIALIZATION #
    ##################################

    def __init__(self):        
        #############################
        # Optimizer hyperparameters #
        #############################
        
        # maximize parameters
        self.verbose = 1 # 0 for no output, 1 for some output printing
        self.random_state = 123 # random seed
        self.init_points = 200 # number of initial points to sample randomly for Bayesian optimization
        self.iterations = 200 # number of iterations to run Bayesian optimization
        
        # Acquisition function        
        # Low kappa means more exploitation for UCB
        # High kappa means more exploration for UCB
        # Low xi means more exploitation for EI and POI
        # High xi means more exploration for EI and POI
        self.acquisitionFunction = UtilityFunction(kind='ucb', kappa=2.576, xi=0, kappa_decay=1, kappa_decay_delay=0)
        #self.acquisitionFunction = UtilityFunction(kind='poi', kappa=2.576, xi=0, kappa_decay=1, kappa_decay_delay=0)
        #self.acquisitionFunction = UtilityFunction(kind='ei', kappa=2.576, xi=0, kappa_decay=1, kappa_decay_delay=0)
        
        # Gaussian process kernel parameters
        self.GP_kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)) # RBF kernel
        #self.GP_kernel = Matern(nu=2.5) # Matern kernel
        self.alpha = 1e-6
        self.normalize_y=True
        self.n_restarts_optimizer=5
        
    ##########################
    # OPTIMIZATION FUNCTIONS #
    ##########################

    def initializeOptimizer(self, lossFunction, param_bounds):
        self.param_bounds = param_bounds
        BO_bounds = param_bounds
        bo_instance = BayesianOptimization(
            f = lossFunction,
            pbounds = BO_bounds, 
            verbose = self.verbose,
            random_state = self.random_state,
            bounds_transformer = None,
            allow_duplicate_points = False
        )
        bo_instance.set_gp_params(
            kernel=self.GP_kernel,
            alpha=self.alpha,
            normalize_y=self.normalize_y,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=self.random_state,
        )
        self.optimizer = bo_instance

    def run(self):
        self.optimizer.maximize(
            init_points = self.init_points, 
            n_iter = self.iterations,   
            acquisition_function=self.acquisitionFunction, 
        )
        
        
    def outputResult(self):
        solution_dict = self.optimizer.max["params"]
        solution_tuple = tuple(solution_dict.items())
        best_solution_loss = self.optimizer.max["target"]
        return solution_dict, solution_tuple, best_solution_loss
    


# In[60]:


BO_instance = BO()
BO_instance.initializeOptimizer(lossFunction, param_bounds)
BO_instance.run()
solution_dict, solution_tuple, best_solution_loss = BO_instance.outputResult()

for param in solution_dict:
    print(f"{param}: {solution_dict[param]}")


# In[35]:


# plt.plot(expt_Disp,expt_Force)
# plt.plot(sim_Disp,sim_Force)


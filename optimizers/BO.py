import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.interpolate import CubicSpline
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from sklearn.gaussian_process.kernels import RBF, Matern # you can try to import other kernels from sklearn as well
import os
from modules.helper import *

class BO():
    
    ##################################
    # OPTIMIZER CLASS INITIALIZATION #
    ##################################

    def __init__(self, info):        
        #############################
        # Optimizer hyperparameters #
        #############################
        self.info = info

        # maximize parameters
        self.verbose = 1 # 0 for no output, 1 for some output printing
        self.random_state = 123 # random seed
        self.init_points = 0 # number of initial points to sample randomly for Bayesian optimization
        self.iterations = 1 # number of iterations to run Bayesian optimization
        
        # Acquisition function        
        # Low kappa means more exploitation for UCB
        # High kappa means more exploration for UCB
        # Low xi means more exploitation for EI and POI
        # High xi means more exploration for EI and POI

        #self.acquisitionFunction = UtilityFunction(kind='ucb', kappa=1, kappa_decay=1, kappa_decay_delay=0)
        #self.acquisitionFunction = UtilityFunction(kind='ucb', kappa=2.576, kappa_decay=1, kappa_decay_delay=0)
        #self.acquisitionFunction = UtilityFunction(kind='ucb', kappa=10, kappa_decay=1, kappa_decay_delay=0)
        #self.acquisitionFunction = UtilityFunction(kind='ei', xi=0)
        #self.acquisitionFunction = UtilityFunction(kind='ei', xi=0.1)
        #self.acquisitionFunction = UtilityFunction(kind='poi', xi=0.001)
        self.acquisitionFunction = UtilityFunction(kind='poi', xi=0.1)

        
        # Gaussian process kernel parameters
        #self.GP_kernel = RBF(length_scale=10, length_scale_bounds=(1e-3, 1e3)) # RBF kernel
        self.GP_kernel = Matern(nu=2.5) # Matern kernel
        # Noise variance regularization hyperparameter: control the Gaussian noise
        self.alpha = 1e-6
        #self.alpha = 0.5
        self.normalize_y=False
        self.n_restarts_optimizer=5
        self.logger = JSONLogger(path="bayesopt_log/logs.log", reset=False)
        
    ##########################
    # OPTIMIZATION FUNCTIONS #
    ##########################
    
    def initializeOptimizer(self, lossFunction, param_bounds, loadingProgress = True):
        self.param_bounds = param_bounds
        self.loadingProgress = loadingProgress
        bo_instance = BayesianOptimization(
            f = lossFunction,
            pbounds = param_bounds, 
            verbose = self.verbose,
            random_state = self.random_state,
            bounds_transformer = None,
            allow_duplicate_points = False,
        )
        bo_instance.set_gp_params(
            kernel=self.GP_kernel,
            alpha=self.alpha,
            normalize_y=self.normalize_y,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=self.random_state
        )
        self.optimizer = bo_instance
        if loadingProgress == False:
            self.optimizer.subscribe(Events.OPTIMIZATION_STEP, self.logger)
        else:
            projectPath = self.info["projectPath"]
            logPath = self.info["logPath"]
            if os.path.exists(f"{projectPath}/optimizers/logs.json"):
                #self.optimizer.subscribe(Events.OPTIMIZATION_STEP, self.logger)
                #load_logs(self.optimizer, logs=["./bayesopt_log/logs.log.json"]);
                load_logs(self.optimizer, logs=[f"{projectPath}/optimizers/logs.json"]);
                printLog("BO optimizer is now aware of {} points.".format(len(self.optimizer.space)), logPath)
                self.optimizer.subscribe(Events.OPTIMIZATION_STEP, self.logger)
        
        
    def run(self):
        self.optimizer.maximize(
            init_points = self.init_points if self.loadingProgress == False else 0, 
            n_iter = self.iterations,   
            acquisition_function=self.acquisitionFunction
        )
    
    # Only use this function when you initialize without loss function
    def suggest(self):
        next_point = self.optimizer.suggest(self.acquisitionFunction)
        return next_point
        
    def outputResult(self):
        solution_dict = self.optimizer.max["params"]
        solution_tuple = tuple(solution_dict.items())
        best_solution_loss = self.optimizer.max["target"]
        return solution_dict, solution_tuple, best_solution_loss
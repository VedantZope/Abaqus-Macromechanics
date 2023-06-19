import sys
import json
import os
import numpy as np
import pandas as pd
import glob
from prettytable import PrettyTable

def printLog(message, logPath):
    with open(logPath, 'a+') as logFile:
        logFile.writelines(message)
    print(message)

def parseBoundsBO(paramInfo):
    paramBounds = {}
    for param in paramInfo:
        paramBounds[param] = (paramInfo[param]['lowerBound'], paramInfo[param]['upperBound'])
    return paramBounds

def is_directory_empty(directory_path):
    return len(os.listdir(directory_path)) == 0
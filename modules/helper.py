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


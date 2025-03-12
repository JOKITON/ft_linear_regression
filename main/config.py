""" config.py """

import os
import sys
from colorama import Fore, Back, Style

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Set the current working directory two directories above
PWD = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))

NUM_ITERATIONS = 600  # Number of iterations to repeat
CONVERGENCE_THRESHOLD = 1e-10  # Threshold for convergence
LEARNING_RATE = 1e-1

RESET_ALL = Fore.RESET + Back.RESET + Style.RESET_ALL

YES_NO = Fore.GREEN + Style.BRIGHT + " (yes" + RESET_ALL
YES_NO += " / " + Fore.RED + Style.BRIGHT + "no): " + RESET_ALL + Style.BRIGHT

# Define paths to datasets
FT_LINEAR_REGRESION_THETAS_PATH = os.path.join(PWD, 'data/thetas.json')
FT_LINEAR_REGRESION_PLOT_PATH = os.path.join(PWD, 'plots/')
FT_LINEAR_REGRESSION_CAR_MILEAGE_TRAIN = os.path.join(PWD, 'data/data.csv')

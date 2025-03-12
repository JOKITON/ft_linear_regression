""" Program that performs linear regression on the car mileage dataset """

import pandas as pd
import config
from colorama import Fore, Back, Style
from df_utils import set_thetas_values
from loss import f_mae, f_mse
from train import forward, comp_gradients
from config import LEARNING_RATE, NUM_ITERATIONS, CONVERGENCE_THRESHOLD
from tqdm import tqdm

RESET_ALL = Fore.RESET + Back.RESET + Style.RESET_ALL

print()

# Load the dataset
df = pd.read_csv(config.FT_LINEAR_REGRESSION_CAR_MILEAGE_TRAIN)

# Normalize mileage values (do this once, not inside the gradient calculation)
mileage = df['km']
mean_mileage = mileage.mean()
std_mileage = mileage.std()

prices = df['price']
mean_price = prices.mean()
std_price = prices.std()

norm_mileage = (mileage - mean_mileage) / std_mileage
norm_price = (prices - mean_price) / std_price

THETA0 = 0
THETA1 = 0

# Gradient descent loop
# for it in range(NUM_ITERATIONS):
with tqdm(
    total=NUM_ITERATIONS,
    desc= Style.BRIGHT + "⌛ Please wait for the results...",
    leave=False) as pbar:
    for it in range(NUM_ITERATIONS):
        pbar.update(1)
        # Compute the gradients
        tmp0, tmp1 = comp_gradients(THETA0, THETA1, norm_mileage, norm_price)

        # Update theta values
        THETA0 -= tmp0
        THETA1 -= tmp1

        # Optional: Check for convergence
        if abs(tmp0) < CONVERGENCE_THRESHOLD and abs(tmp1) < CONVERGENCE_THRESHOLD:
            mse = f_mse(norm_mileage, norm_price, THETA0, THETA1)
            print("\tIteration " + Fore.LIGHTWHITE_EX
                + Style.BRIGHT + f"{it}" + RESET_ALL
                + ": MSE [" + Style.BRIGHT + f"{mse:.3f}]" + RESET_ALL)
            print( Fore.GREEN + Style.BRIGHT
                + f"\nConverged after {it+1} iterations!" + RESET_ALL + "\n")
            break

        if (it % 50 == 0):
            if (it == 0):
                print(RESET_ALL)
            mse = f_mse(norm_mileage, norm_price, THETA0, THETA1)
            print("\tIteration " + Fore.LIGHTWHITE_EX
                + Style.BRIGHT + f"{it}" + RESET_ALL
                + ": MSE [" + Style.BRIGHT + f"{mse:.3f}]" + RESET_ALL)

""" preds = forward(THETA0, THETA1, norm_mileage)
preds = (preds * prices.std()) + prices.mean()
print(preds) """

# Output the final results
print("🧮 Results:" + RESET_ALL)

print(Style.BRIGHT + "\t✴️ theta0: " + Fore.LIGHTBLUE_EX
      + Style.BRIGHT + f"{THETA0}" + RESET_ALL)

print(Style.BRIGHT + "\t✴️ theta1: " + Fore.LIGHTCYAN_EX
      + Style.BRIGHT + f"{THETA1}" + RESET_ALL)

print(Style.DIM + "\n📥 Theta values have been saved in "
        + config.FT_LINEAR_REGRESION_THETAS_PATH + RESET_ALL)

set_thetas_values(THETA0, THETA1)

""" Program to predict car prices using a linear regression model """

import pandas as pd
import config
from config import RESET_ALL, YES_NO
from colorama import Fore, Back, Style
from plot_utils import crt_diverse_df, crt_plot, crt_diverse_plot
from df_utils import get_thetas_values, display_precision
import numpy as np

# Get theta values from thetas.json
THETA0, THETA1 = get_thetas_values()

def normalize_data():
    df = pd.read_csv(config.FT_LINEAR_REGRESSION_CAR_MILEAGE_TRAIN)
    mileage = np.array(df['km'])
    prices = np.array(df['price'])
    
    mileage_mean = mileage.mean()
    mileage_std = mileage.std()

    price_mean = prices.mean()
    price_std = prices.std()

    normalized_mileage = (mileage - mileage_mean) / mileage_std
    normalized_price = (prices - price_mean) / price_std
    
    return normalized_mileage, normalized_price, mileage, prices

def make_one_prediction(mileage, prices, to_predict):
    """ Make a prediction for a single mileage value """

    normalized_x = (to_predict - mileage.mean()) / mileage.std()

    price = THETA0 + (THETA1 * normalized_x)
    # Denormalize the price
    real_price = (price * prices.std()) + prices.mean()

    print(
        f"\t📊 Predicted price for a car with {to_predict} km: "
            + Fore.GREEN + Style.BRIGHT + f"{real_price:.2f}\n")

    return real_price

def make_all_predictions(norm_mileage, norm_prices, prices):
    """ Make predictions for all mileage values in the dataset """

    # Initialize a list for predicted prices (un-normalized)
    ret_predicted_prices = []
    normalized_pred_prices = []

    # Loop over each mileage value to make predictions
    for mile_sample in norm_mileage:
        # Predict price using the normalized mileage
        price = THETA0 + (THETA1 * mile_sample)

        # Denormalize the price
        real_price = (price * prices.std()) + prices.mean()

        ret_predicted_prices.append(real_price)
        normalized_pred_prices.append(price)

    # Display MSE, MAE and R^2
    print("\n📊 Displaying results with " + Style.BRIGHT + "normalized" + RESET_ALL + " data:")
    display_precision(norm_prices, normalized_pred_prices)
    # Same thing but with un-normalized data
    print("\n📊 Displaying results with " + Style.BRIGHT + "un-normalized" + RESET_ALL + " data:")
    display_precision(prices, ret_predicted_prices)

    return ret_predicted_prices

def plot_all_predictions(mileage, prices, arg_predicted_prices):
    """ Plot actual vs predicted prices for all mileage values """

    check = 0
    while check == 0:
        str1 = input("└─> Do you want to create a plot of the data and save it locally?" + YES_NO)
        print(RESET_ALL)
        if str1 == "yes":
            crt_plot(mileage, prices, arg_predicted_prices, "all")
            while check == 0:
                str2 = input(
                    "\n└─> Do you want to create some plots with different km ranges?" + YES_NO)
                print(RESET_ALL)
                if str2 == "yes":
                    crt_diverse_plot(mileage, prices, arg_predicted_prices)
                    check = 1
                    break
                if str2 == "no":
                    check = 1
                    break
                print("Invalid input. Please try again.\n")
                continue
            break
        if str1 == "no":
            break
        print("Invalid input. Please try again.\n")
        continue

def main():
    norm_mileage, norm_prices, mileage, prices = normalize_data()
    while 1:
        main_str1 = input(
            "└─> Do you want to predict" + Style.BRIGHT + Fore.LIGHTMAGENTA_EX
                + " all the prices" + RESET_ALL + " based on the dataset" + YES_NO)
        print(RESET_ALL)
        if main_str1 == "yes":
            predicted_prices = make_all_predictions(norm_mileage, norm_prices, prices)
            plot_all_predictions(mileage, prices, predicted_prices)
            break
        if main_str1 == "no":
            break
        print("Invalid input. Please try again.\n")
        continue

    while 1:
        main_str2 = input(
            RESET_ALL + "\n└─> Please, input a mileage (" + Fore.LIGHTWHITE_EX + Style.BRIGHT
                + "only numbers" + RESET_ALL + "): ")
        if main_str2.isdigit():
            if int(main_str2) < 0 or int(main_str2) > 1000000:
                print(
                    Fore.RED + Style.DIM + "Invalid input. Please try again.")
                continue
            to_predict = int(main_str2)
            make_one_prediction(mileage, prices, to_predict)
            main_str3 = input(
                Fore.RESET + Back.RESET + Style.RESET_ALL
                    + "└─> Do you want to predict the price for another mileage?" + YES_NO)
            if main_str3 == "yes":
                continue
            if main_str3 == "no":
                print(
                    Fore.GREEN + Style.BRIGHT
                        + "👋  Thank you for using my program. Goodbye fellow 42 Coder!\n")
                break
            print(Fore.RED + Style.DIM + "Invalid input. Please try again.")
            continue
        print(Fore.RED + Style.DIM + "Invalid input. Please try again.")
        continue

main()

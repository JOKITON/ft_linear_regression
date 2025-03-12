""" Python program to print the normalized values of mileage and price """

import sys
import json
import config
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def get_paths():
    """ Reset the current working directory """
    print(sys.path)

def get_thetas_values() :
    """ Retrieve theta values from thetas.json """
    try:
        with open(
            config.FT_LINEAR_REGRESION_THETAS_PATH, "r", encoding="utf-8"
            ) as read_file:
            data = json.load(read_file)
    except FileNotFoundError:
        print("Error: thetas.json not found")
    return data["THETA0"], data["THETA1"]

def set_thetas_values(theta0, theta1) :
    """ Retrieve theta values from thetas.json """
    try:
        with open(
            config.FT_LINEAR_REGRESION_THETAS_PATH, "r", encoding="utf-8"
            ) as read_file:
            data = json.load(read_file)
        data['THETA0'] = theta0
        data['THETA1'] = theta1
        # Save the updated thetas back to the json file
        with open(config.FT_LINEAR_REGRESION_THETAS_PATH, "w", encoding="utf-8"
            ) as file:
            json.dump(data, file)
    except FileNotFoundError:
        print("Error: thetas.json not found")
    return data["THETA0"], data["THETA1"]

def denormalize_parameters(theta0, theta1, mean, std):
    """ Adjust theta0 and theta1 to work directly with real mileage values. """
    theta1_prime = theta1 / std
    theta0_prime = theta0 - (theta1 * mean / std)
    return theta0_prime, theta1_prime


def print_normalized_val(df):
    """ Normalize mileage values (do this once, not inside the gradient calculation) """
    mean_mileage = df['km'].mean()
    sigma_mileage = df['km'].std()
    print("Sigma (Standard deviation of mileage):", sigma_mileage)
    df['km'] = (df['km'] - mean_mileage) / sigma_mileage

    mean_price = df['price'].mean()
    sigma_price = df['price'].std()
    df['price'] = (df['price'] - mean_price) / sigma_price

    print(df['km'])
    print(df['price'])

def display_results(df, predicted_prices):
    """ Display results of the linear regression model """
    for mileage, price in zip(df['km'], predicted_prices):
        print(f"Mileage: {mileage} km, Predicted Price: {price:.2f}")

def display_precision(prices, predicted_prices):
    """ Display the precision of the linear regression model """
    mse = mean_squared_error(prices, predicted_prices)
    mae = mean_absolute_error(prices, predicted_prices)
    r2 = r2_score(prices, predicted_prices)

    print(f"📉 MSE: {mse:.2f},\n📈 MAE: {mae:.2f},\n📋 R^2: {r2:.2f}\n")

""" Functions to create plots for the linear regression project. """

import matplotlib.pyplot as plt
import config
import numpy as np
from df_utils import get_thetas_values
import numpy as np

def crt_plot(mileage, prices, predicted_prices, mile_frame, regression_line=None):
    """ Create a plot of actual vs predicted prices and the regression line """

    if mile_frame != 'all':
        file_print_bool = 1
    else:
        file_print_bool = 0
    theta0, theta1 = get_thetas_values()

    # Scatter plot of actual and predicted prices
    plt.scatter(
        mileage,
        prices,
        color='blue',
        label='Actual Prices (<= ' + mile_frame + ' km)')
    plt.scatter(
        mileage,
        predicted_prices,
        color='red',
        label='Predicted Prices (<= ' + mile_frame + ' km)')

    # Generate a range of mileage values
    mileage_range = np.linspace(22899.0, 240000.0, 1000)

    if (regression_line is None):
        # Normalize the mileage range for the regression function
        normalized_mileage = (mileage_range - mileage.mean()) / mileage.std()
        # Compute the regression line using normalized mileage
        regression_line = theta0 + theta1 * normalized_mileage
        if (regression_line.sum() != 0):
            regression_line = (regression_line * prices.std()) + prices.mean()

    # Plot the regression line
    plt.plot(
        mileage_range,
        regression_line,
        color='green',
        label=f'Prediction Line: $\\theta_0$={theta0:.2f}, $\\theta_1$={theta1:.2f}')

    # Force consistent Y-axis limits
    ylin_min = 2851
    ylin_max = 8790
    plt.ylim(ylin_min, ylin_max)

    # Plot settings
    plt.xlabel('Mileage (km)')
    plt.ylabel('Price')
    plt.legend()
    plt.title('Actual vs Predicted Prices')

    # Save the plot to a file
    plt.savefig(
        config.FT_LINEAR_REGRESION_PLOT_PATH + 'actual_vs_predicted_prices_' + mile_frame + '.png')
    print(
        "📥 Plot saved as 'actual_vs_predicted_prices_" + mile_frame + ".png'")

    if file_print_bool == 0:
        print(
            "📩 File location : " + config.FT_LINEAR_REGRESION_PLOT_PATH + "")
        file_print_bool = 1

    # Clear the current plot to avoid overlapping
    plt.clf()

def crt_diverse_df(mileage, prices, predicted_prices, range):
    """ Create diverse plots based on mileage ranges """
    ret_mileage = np.array([])
    ret_price = np.array([])
    ret_predicted = np.array([])
    for mileage, price, predicted_price in zip(mileage, prices, predicted_prices):
        if mileage <= range and mileage >= range - 100000:
            ret_mileage = np.append(ret_mileage, mileage)
            ret_price = np.append(ret_price, price)
            ret_predicted = np.append(ret_predicted, predicted_price)
    return ret_mileage, ret_price, ret_predicted

def crt_diverse_plot(mileage, prices, predicted):
    """ Create diverse plots on mileage ranges going from 0-100k, 100-200k, 200-300k """

    theta0, theta1 = get_thetas_values()
    mileage_range = np.linspace(np.min(mileage), np.max(mileage), 1000)
    # Normalize the mileage range for the regression function
    normalized_mileage = (mileage_range - mileage.mean()) / mileage.std()
    # Compute the regression line using normalized mileage
    regression_line = theta0 + theta1 * normalized_mileage
    regression_line = (regression_line * prices.std()) + prices.mean()

    sp_mileage, sp_prices, sp_predicted = crt_diverse_df(mileage, prices, predicted, 100000)
    crt_plot(
           sp_mileage, sp_prices, sp_predicted, '100k', regression_line)
    sp_mileage, sp_prices, sp_predicted = crt_diverse_df(mileage, prices, predicted, 200000)
    crt_plot(
           sp_mileage, sp_prices, sp_predicted, '200k', regression_line)
    sp_mileage, sp_prices, sp_predicted = crt_diverse_df(mileage, prices, predicted, 300000)
    crt_plot(
           sp_mileage, sp_prices, sp_predicted, '300k', regression_line)

import config
from config import LEARNING_RATE

def forward(theta0, theta1, mileage):
    """ Main function to predict the price of a car """
    return theta0 + (theta1 * mileage)

def comp_gradients(theta0, theta1, mileage, price):
    """ Compute the gradients of the cost function """
    m = len(mileage)
    sum0 = 0
    sum1 = 0
    for i in range(m):
        sum0 += forward(
            theta0, theta1, mileage[i]) - price[i]

        sum1 += (forward(
            theta0, theta1, mileage[i]) - price[i]) * mileage[i]
    ret_tmp0, ret_tmp1 = sum0 / m, sum1 / m
    ret_tmp0 = LEARNING_RATE * ret_tmp0
    ret_tmp1 = LEARNING_RATE * ret_tmp1
    return ret_tmp0, ret_tmp1

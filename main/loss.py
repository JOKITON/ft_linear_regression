from train import forward

def f_mse(mileage, price, theta0, theta1):
    """ Compute the mean squared error """
    predictions = [forward(theta0, theta1, x) for x in mileage]
    ret_mse = sum((price - predictions) **2) / len(mileage)
    return ret_mse

def f_mae(mileage, price, theta0, theta1):
    """ Compute the mean squared error """
    predictions = [forward(theta0, theta1, x) for x in mileage]
    ret_mae = sum((price - predictions)) / len(mileage)
    return ret_mae

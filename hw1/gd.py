import numpy as np

def cost_function( x, y, theta0, theta1 ):
    """Compute the squared error cost function

    Inputs:
    x        vector of length m containing x values
    y        vector of length m containing y values
    theta_0  (scalar) intercept parameter
    theta_1  (scalar) slope parameter

    Returns:
    cost     (scalar) the cost
    """

    cost = 1.0

    ##################################################
    # TODO: write code here to compute cost correctly
    ##################################################
    sub = np.square(((np.dot(x, theta1)+theta0)-y))
    cost = np.sum(sub)/2
    return cost


def gradient(x, y, theta_0, theta_1):
    """Compute the partial derivative of the squared error cost function

    Inputs:
    x          vector of length m containing x values
    y          vector of length m containing y values
    theta_0    (scalar) intercept parameter
    theta_1    (scalar) slope parameter

    Returns:
    d_theta_0  (scalar) Partial derivative of cost function wrt theta_0
    d_theta_1  (scalar) Partial derivative of cost function wrt theta_1
    """

    d_theta_0 = 0.0
    d_theta_1 = 0.0

    ##################################################
    # TODO: write code here to compute partial derivatives correctly
    ##################################################
    sub = (np.dot(x, theta_1)+theta_0)-y
    sub_2 = x.T.dot(sub)
    d_theta_0 = np.sum(sub)
    d_theta_1 = np.sum(sub_2)

    return d_theta_0, d_theta_1 # return is a tuple
from gradient_descent import *
from stochastic_gradient_descent import *

import numpy as np


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    loss, w = gradient_descent(y, tx, initial_w, max_iters, gamma)
    return (w, loss)

def least_squared_SGD(y, tx, initial_w, max_iters, gamma):
    loss, w = stochastic_gradient_descent(y, tx, initial_w, 1, max_iters, gamma)
    return (w, loss)


def least_squares(y, tx):
    tx_t = tx.transpose()
    w = np.linalg.solve(tx_t @ tx, tx_t @ y)
    loss = compute_loss(y, tx, w)
    return loss, w


def ridge_regression(y, tx, lambda_):
    tx_t = tx.transpose()
    d = tx.shape[1]
    x_T_x =  tx_t @ tx + lambda_ * np.diag(np.ones(d))
    w = np.linalg.solve(x_T_x, tx_t @ y)
    loss = compute_loss(y, tx, w)
    return loss, w

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


def sigmoid(x):
    return 1/(1+np.exp(-x))


def calculate_loss_logistic(y, tx, w):
    sig_fitted = sigmoid(tx @ w)
    losses =  y* np.log(sig_fitted) + (1-y) * np.log(1 - sig_fitted)
    
    return - losses.sum()


def calculate_gradient_logistic(y, tx, w):
    sig_fitted = sigmoid(tx @ w)
    return tx.transpose() @ ( sig_fitted - y[:, None] )


def learning_by_gradient_descent_logistic(y, tx, w, gamma):
 
    gradient = calculate_gradient_logistic(y, tx, w)
    w = w-gamma*gradient 
    loss = calculate_loss_logistic(y, tx, w)
    return loss, w


def logistic_regression_gradient_descent_demo(y, x, max_iter=1000, threshold=1e-8, gamma = 0.0001):
 
    w = np.zeros((x.shape[1], 1))
    losses = []
    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_gradient_descent_logistic(y, x, w, gamma)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w 

    
    
    
    
    
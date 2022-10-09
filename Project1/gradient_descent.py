from costs import *

import numpy as np

def compute_gradient(y, tx, w):
    fitted = tx @ w
    e = y - fitted
    n = fitted.shape[0]
    gradient = -1/n  * tx.transpose() @ e
    return gradient 


def gradient_descent(y, tx, initial_w, max_iters, gamma):

    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y,tx, w)
        w = w - gradient * gamma 
        loss = compute_loss(y,tx, w)
    return loss, w
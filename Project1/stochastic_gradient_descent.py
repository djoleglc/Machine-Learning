import numpy as np
# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""
from helpers import *
from costs import *

def compute_stoch_gradient(y, tx, w):
    fitted = tx @ w
    e = y - fitted
    n = fitted.shape[0]
    gradient = -1/n  * tx.transpose() @ e
    return gradient


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""

    w = initial_w
    for n_iter in range(max_iters):
        for yb, txb in batch_iter(y,tx, batch_size, int(len(y)/batch_size)):
            
            gradient = compute_stoch_gradient(yb, txb, w)
            w = w - gradient * gamma
            loss = compute_loss(yb,txb, w)
        
    return loss, w